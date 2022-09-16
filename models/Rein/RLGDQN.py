import time
from models.embedder import embedder_single
from evaluate import accuracy
from models.Layers import Env_Net_RLG, act_layer, GNN_Model
import numpy as np
import random as random
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from collections import defaultdict

np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

def get_A_r(adj, r):
    adj_label = adj
    for i in range(r - 1):
        adj_label = adj_label @ adj

    return adj_label

def Ncontrast(x_dis, adj_label, tau = 1):
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def KDloss_0( y, teacher_scores):
    T = 4
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T ** 2) / y.shape[0]
    return l_kl

def get_log_likelihood(_log_p, pi):
    """	args:
        _log_p: (batch, city_t, city_t)
        pi: (batch, city_t), predicted tour
        return: (batch)
    """
    log_p = torch.gather(input=_log_p, dim=2, index=pi[:, :, None]).squeeze(-1)

    # index_truncation = torch.max(pi, 1)[1]
    # mask = torch.zeros((log_p.shape[0], log_p.shape[1] + 1), dtype=log_p.dtype, device=log_p.device)
    # mask[(torch.arange(log_p.shape[0]), index_truncation + 1)] = 1
    # mask = mask.cumsum(dim=1)[:, :-1]  # remove the superfluous column
    # log_p = log_p * (1. - mask)  # use mask to zero after each column
    # return torch.div(torch.sum(log_p.squeeze(-1), 1), index_truncation + 1)

    return torch.mean(log_p.squeeze(-1), 1)

class Categorical(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_p):
        return torch.multinomial(log_p.exp(), 1).long().squeeze(1)

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

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # scores = torch.where(scores > 1.1 * scores.mean(), scores, torch.zeros_like(scores))

    if mask is not None:
        mask = mask.unsqueeze(0)
        scores = scores.masked_fill(mask, -1e9)

    scores = F.softmax(scores, dim=-1)
    # scores = torch.where(scores > scores.mean(), scores, torch.zeros_like(scores))
    # scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention_new(nn.Module):
    def __init__(self, heads, d_model_in, d_model_out, dropout=0.1):
        super().__init__()

        self.d_model = d_model_out
        self.d_k = d_model_out // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model_in, d_model_out)
        self.v_linear = nn.Linear(d_model_in, d_model_out)
        self.k_linear = nn.Linear(d_model_in, d_model_out)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model_out, d_model_out)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, self.h, self.d_k)
        q = self.q_linear(q).view(bs, self.h, self.d_k)
        v = self.v_linear(v).view(bs, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 0)
        q = q.transpose(1, 0)
        v = v.transpose(1, 0)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(0, 1).contiguous().view(bs, self.d_model)
        output = self.out(concat)

        return output

class Act_Model(nn.Module):
    def __init__(self, n_in ,cfg = None, batch_norm=True, act='elu', dropout = 0.1, final_mlp = 0, heads = 4):
        super(Act_Model, self).__init__()

        self.dropout = dropout
        self.bat = batch_norm
        self.act = act
        self.final_mlp = final_mlp > 0
        self.cfg = cfg
        self.layer_num = len(cfg)
        in_channels = n_in
        self.norm_0 = Norm(in_channels)
        self.mlp_1 = nn.Linear(in_channels, cfg[0])
        self.norm_1 = Norm(cfg[0])
        self.norm_2 = Norm(cfg[0])
        self.trans_1 = MultiHeadAttention_new(heads, cfg[0],cfg[0], dropout=dropout)
        self.ff_1 = FeedForward(cfg[0], dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.selecter = Categorical()

        # for m in self.modules():
        #     self.weights_init(m)
    # def weights_init(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight.data)
    #         if m.bias is not None:
    #             m.bias.data.fill_(0.0)
    #     if isinstance(m, nn.Sequential):
    #         for mm in m:
    #             try:
    #                 torch.nn.init.xavier_uniform_(mm.weight.data)
    #             except:
    #                 pass


    def forward(self, x_input, mask, q_idx, k_idx):
        pi_list= []
        log_ps = []

        x_input = self.norm_0(x_input)
        x_1 = self.dropout_1(self.norm_1(F.elu(self.mlp_1(x_input))))

        x_2 = self.trans_1(x_1, x_1, x_1, mask) + x_1
        x_2 = self.norm_2(F.elu(x_2))

        x_3 = x_2# self.dropout_2(self.ff_1(x_2)) + x_1

        q_embedding = x_3[q_idx].unsqueeze(1) #[Batch, 1, dim]
        k_embedding = x_3[k_idx].unsqueeze(0).repeat(q_embedding.size(0), 1, 1).transpose(-2, -1) #[Batch, dim, train_num]
        logits = torch.bmm(q_embedding,k_embedding).squeeze()
        logits = logits * (torch.sum(logits, 1)**(-1)).unsqueeze(-1)
        log_p = torch.softmax(logits, dim=-1)
        neibor_select = self.selecter(log_p)  # 通过学习得到的动作概率抽样选择最终动作
        pi_list.append(neibor_select)
        log_ps.append(log_p)

        ll = torch.gather(input= log_ps[0], dim = 1, index=pi_list[0][:,None]).squeeze()   #一系列动作的的概率和用于反向传播
        return pi_list[0], ll #因为第一个为stop，所有所有位置需要-1

class RLGDQN(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.cfg = args.cfg
        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.cfg.append(nb_classes)
        self.graph_org_torch = self.adj_list[0]

        self.action_num = 5
        self.max_episodes = 325
        self.max_timesteps = 10

        self.data_split = zip(self.idx_train.cpu(), self.idx_val.cpu(), self.idx_test.cpu())
        self.env_model = GNN_Model(self.args.ft_size, cfg=self.cfg[1:], final_mlp = 0, gnn = self.args.gnn, dropout=self.args.random_aug_feature).to(self.args.device)
        self.act_model = Act_Model(self.args.ft_size, cfg=self.cfg, final_mlp = 0, dropout=self.args.random_aug_feature).to(self.args.device)
        self.act_optimizer = torch.optim.Adam(self.act_model.parameters(), lr=0.001)

    def pre_training(self):

        features = self.features.to(self.args.device)
        # graph_org = self.dgl_graph.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)
        print("Started training...")

        optimiser = torch.optim.Adam(self.env_model.parameters(), lr=0.01, weight_decay=5e-4)
        xent = nn.CrossEntropyLoss()
        train_lbls = self.labels[self.idx_train]
        val_lbls = self.labels[self.idx_val]
        test_lbls = self.labels[self.idx_test]

        cnt_wait = 0
        best = 1e-9
        output_acc = 1e-9
        stop_epoch = 0

        start = time.time()
        totalL = []
        # features = F.normalize(features)
        Last_embedding = None
        for epoch in range(self.args.nb_epochs):
            self.env_model.train()
            optimiser.zero_grad()

            embeds = self.env_model(graph_org_torch, features)
            embeds_preds = torch.argmax(embeds, dim=1)

            train_embs = embeds[self.idx_train]
            val_embs = embeds[self.idx_val]
            test_embs = embeds[self.idx_test]

            loss = F.cross_entropy(train_embs, train_lbls)

            loss.backward()
            totalL.append(loss.item())
            optimiser.step()

            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0:
                self.env_model.eval()
                # A_a, X_a = process.RA(graph_org.cpu(), features, 0, 0)
                # A_a = A_a.add_self_loop().to(self.args.device)
                embeds = self.env_model(graph_org_torch, features)
                Last_embedding = embeds.detach()
                val_acc = accuracy(embeds[self.idx_val], val_lbls)
                test_acc = accuracy(embeds[self.idx_test], test_lbls)

                stop_epoch = epoch
                if val_acc > best:
                    best = val_acc
                    output_acc = test_acc.item()
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if cnt_wait == self.args.patience:
                    break
            ################END|Eval|###############

        training_time = time.time() - start
        print("\t[Classification] ACC: {:.4f} | stop_epoch: {:}| training_time: {:.4f} ".format(
            output_acc, stop_epoch, training_time))
        return Last_embedding

    def act_training(self, env_embedding):
        self.act_model.train()
        dis_embedding = get_feature_dis(env_embedding)
        features = self.features.to(self.args.device)
        graph_org = self.dgl_graph.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)
        print("Started training...")
        idx_test_val = list(self.idx_test.cpu().numpy()) + list(self.idx_val.cpu().numpy())
        idx_train = list(self.idx_train.cpu().numpy())
        batch_size = self.args.batch
        labeled_size = self.idx_train.size(0)
        q_idx = range(batch_size)
        k_idx = range(batch_size, batch_size + labeled_size)
        mask = torch.zeros([batch_size + labeled_size, batch_size + labeled_size])
        mask[:, : batch_size] = 1.0
        mask = mask.bool().to(self.args.device)
        for epoch in range(self.args.nb_epochs):
            batch_idx = random.sample(idx_test_val, batch_size)
            batch_input = features[batch_idx+idx_train]

            pi_list, ll = self.act_model(batch_input, mask, q_idx, k_idx)
            real_reward = torch.diag(dis_embedding[batch_idx][:,pi_list]).detach()



            act_loss = -(100*torch.exp(real_reward) * ll).mean()        #*10是因为loss太小   #常规的AC网络， 这里应该是max_pi sum_i{Q*p_i} 这里的Q从reward的期望换成了loss
            self.act_optimizer.zero_grad()
            act_loss.backward()
            # nn.utils.clip_grad_norm_(self.act_model.parameters(), max_norm=1., norm_type=2)
            self.act_optimizer.step()
            neibor_acc = self.labels[batch_idx].eq(self.labels[pi_list]).double().sum()/  len(self.labels[pi_list])
            print("Loss: {:.4f}, ACC: {:.1f}".format(act_loss.item(),neibor_acc))





        return None

    def training(self):
        env_embedding = self.pre_training()
        self.act_training(env_embedding)
        return None

