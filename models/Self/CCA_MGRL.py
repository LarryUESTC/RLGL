import time
import os
from models.embedder import embedder
from tqdm import tqdm
from evaluate import evaluate
from models.Layers import make_mlplayers
#from models.SUGRL_Fast import SUGRL_Fast
import numpy as np
import random as random
import torch.nn.functional as F
import torch
import torch.nn as nn
import copy
from models.Layers import act_layer
from torch.nn.functional import cosine_similarity

np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_A_r(adj, r):
    adj_label = adj
    for i in range(r - 1):
        adj_label = adj_label @ adj
    return adj_label

def local_preserve(x_dis, adj_label, tau = 1.0):
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def CCASSL(z_list, N, I_target, num_view):
    if num_view == 2:
        embeding_a = z_list[0]
        embeding_b = z_list[1]
        embeding_a = (embeding_a - embeding_a.mean(0)) / embeding_a.std(0)
        embeding_b = (embeding_b - embeding_b.mean(0)) / embeding_b.std(0)
        c1 = torch.mm(embeding_a.T, embeding_a) / N
        c2 = torch.mm(embeding_b.T, embeding_b) / N
        loss_c1 = (I_target - c1).pow(2).mean() + torch.diag(c1).mean()
        loss_c2 = (I_target - c2).pow(2).mean() + torch.diag(c2).mean()
        loss_C = loss_c1 + loss_c2
        loss_simi = cosine_similarity(embeding_a, embeding_b, dim=-1).mean()
    else:
        embeding_a = z_list[0]
        embeding_b = z_list[1]
        embeding_c = z_list[2]
        embeding_a = (embeding_a - embeding_a.mean(0)) / embeding_a.std(0)
        embeding_b = (embeding_b - embeding_b.mean(0)) / embeding_b.std(0)
        embeding_c = (embeding_c - embeding_c.mean(0)) / embeding_c.std(0)
        c1 = torch.mm(embeding_a.T, embeding_a) / N
        c2 = torch.mm(embeding_b.T, embeding_b) / N
        c3 = torch.mm(embeding_c.T, embeding_c) / N
        loss_c1 = (I_target - c1).pow(2).mean() + torch.diag(c1).mean()
        loss_c2 = (I_target - c2).pow(2).mean() + torch.diag(c2).mean()
        loss_c3 = (I_target - c3).pow(2).mean() + torch.diag(c3).mean()
        loss_C = loss_c1 + loss_c2 + loss_c3
        loss_simi = cosine_similarity(embeding_a, embeding_b, dim=-1).mean() + cosine_similarity(embeding_a, embeding_c, dim=-1).mean() + cosine_similarity(embeding_b, embeding_c, dim=-1).mean()
    return loss_C, loss_simi

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

def local_make_mlplayers(in_channel, cfg, batch_norm=False, out_layer=None):
    layers = []
    in_channels = in_channel
    layer_num = len(cfg)
    act = 'gelu'
    for i, v in enumerate(cfg):
        out_channels = v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm and i != (layer_num - 1):
            layers += [mlp, Norm(out_channels), act_layer(act)]
        elif i != (layer_num - 1):
            layers += [mlp, act_layer(act)]
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)  # , result

class CCA_MGRL_model(nn.Module):
    def __init__(self, n_in , view_num, cfg = None, dropout = 0.2,sparse = True):
        super(CCA_MGRL_model, self).__init__()
        self.view_num = view_num
        MLP = local_make_mlplayers(n_in, cfg, batch_norm = False)
        self.MLP_list = get_clones(MLP, self.view_num)
        self.dropout = dropout
        self.A = None
        self.sparse = sparse
        self.cfg = cfg

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, adj_list=None):
        if self.A is None:
            self.A = adj_list

        view_num = len(adj_list)

        x_list = [F.dropout(x, self.dropout, training=self.training) for i in range(view_num)]

        z_list = [self.MLP_list[i](x_list[i]) for i in range(view_num)]

        s_list = [get_feature_dis(z_list[i]) for i in range(view_num)]

        # simple average
        z_unsqu = [z.unsqueeze(0) for z in z_list]
        z_fusion = torch.mean(torch.cat(z_unsqu), 0)

        return z_list, s_list, z_fusion

    def embed(self,  x , adj_list=None ):
        view_num = len(adj_list)

        x_list = [x for i in range(view_num)]

        z_list = [self.MLP_list[i](x_list[i]) for i in range(view_num)]

        # s_list = [get_feature_dis(z_list[i]) for i in range(view_num)]

        # simple average
        z_unsqu = [z.unsqueeze(0) for z in z_list]
        z_fusion = torch.mean(torch.cat(z_unsqu), 0)


        return z_fusion.detach()


class CCA_MGRL(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.cfg = args.cfg
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):

        features = self.features.to(self.args.device)
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]

        adj_label_list = [get_A_r(adj, self.args.A_r) for adj in adj_list]

        N = features.size(0)
        I_target = torch.tensor(np.eye(self.cfg[-1])).to(self.args.device)
        # print("Started training...")

        model = CCA_MGRL_model(self.args.ft_size, self.args.view_num, cfg=self.cfg, dropout=self.args.dropout).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr)

        model.train()
        start = time.time()

        # for epoch in tqdm(range(self.args.nb_epochs)):
        for epoch in range(self.args.nb_epochs):
            model.train()
            optimiser.zero_grad()
            z_list, s_list, z_fusion = model(features, adj_list)

            loss_local = 0
            for i in range(self.args.view_num):
                loss_local += 1 * local_preserve(s_list[i], adj_label_list[i], tau=self.args.tau)


            loss_C, loss_simi = CCASSL(z_list, N, I_target, self.args.view_num)
            #loss = (1 - loss_simi + loss_C * self.args.w_c) * self.args.w_s + loss_local * self.args.w_l
            loss = (1 - loss_simi + loss_C * self.args.w_c)+ loss_local * self.args.w_l

            loss.backward()
            optimiser.step()

            # if epoch % 100 == 0 and epoch > 0:
            #     model.eval()
            #     hf = model.embed(features, adj_list)
            #     acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, k1, k2, st = evaluate(
            #         hf, self.idx_train, self.idx_val, self.idx_test, self.labels,
            #         seed=self.args.seed, epoch=self.args.test_epo, lr=self.args.test_lr)

        training_time = time.time() - start
        # print("training time:{}s".format(training_time))
        # print("Evaluating...")
        model.eval()
        hf = model.embed(features, adj_list)
        acc, acc_std, macro_f1,macro_f1_std, micro_f1,micro_f1_std,k1, k2, st = evaluate(
            hf, self.idx_train, self.idx_val, self.idx_test, self.labels,
            seed=self.args.seed, epoch=self.args.test_epo,lr=self.args.test_lr)
        return acc, acc_std, macro_f1,macro_f1_std, micro_f1,micro_f1_std,k1, k2, st,training_time

