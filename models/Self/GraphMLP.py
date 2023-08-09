import time
import os
from models.embedder import embedder_single
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
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
import math
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

def target_distribution(q, time = 2):
    weight = q ** time / q.sum(0) #todo
    # return q
    return (weight.T / weight.sum(1)).T

def full_attention_conv(qs, ks, vs, kernel, output_attn=False):
    '''
    qs: query tensor [N, H, M]
    ks: key tensor [L, H, M]
    vs: value tensor [L, H, D]

    return output [N, H, D]
    '''
    if kernel == 'simple':
        # normalize input
        qs = qs / torch.norm(qs, p=2) # [N, H, M]
        ks = ks / torch.norm(ks, p=2) # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs) # [N, H, D]
        all_ones = torch.ones([vs.shape[0]]).to(vs.device)
        vs_sum = torch.einsum("l,lhd->hd", all_ones, vs) # [H, D]
        attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1) # [N, H, D]

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer # [N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks) / attention_normalizer.unsqueeze(2) # [N, L, H]

    elif kernel == 'sigmoid':
        # numerator
        attention_num = torch.sigmoid(torch.einsum("nhm,lhm->nlh", qs, ks))  # [N, L, H]

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        attention_normalizer = torch.einsum("nlh,l->nh", attention_num, all_ones)
        attention_normalizer = attention_normalizer.unsqueeze(1).repeat(1, ks.shape[0], 1)  # [N, L, H]

        # compute attention and attentive aggregated results
        attention = attention_num / attention_normalizer
        attn_output = torch.einsum("nlh,lhd->nhd", attention, vs)  # [N, H, D]

    if output_attn:
        return attn_output, attention
    else:
        return attn_output

def gcn_conv(x, edge_index, edge_weight):
    N, H = x.shape[0], x.shape[1]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    gcn_conv_output = []
    if edge_weight is None:
        value = torch.ones_like(row) * d_norm_in * d_norm_out
    else:
        value = edge_weight * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    for i in range(x.shape[1]):
        gcn_conv_output.append( matmul(adj, x[:, i]) )  # [N, D]
    gcn_conv_output = torch.stack(gcn_conv_output, dim=1) # [N, H, D]
    return gcn_conv_output

class DIFFormerConv(nn.Module):
    '''
    one DIFFormer layer
    '''
    def __init__(self, in_channels,
               out_channels,
               num_heads,
               kernel='simple',
               use_graph=False,
               use_weight=True):
        super(DIFFormerConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel = kernel
        self.use_graph = use_graph
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, self.kernel, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(query,key,value,self.kernel) # [N, H, D]

        # use input graph for gcn conv
        if self.use_graph:
            final_output = attention_output + gcn_conv(value, edge_index, edge_weight)
        else:
            final_output = attention_output
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output

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
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    # x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    # x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    # x_sum = x_sum @ x_sum.T
    # x_dis = x_dis*(x_sum**(-1)+ 1e-7)
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

def forward_label(self, q_i, q_j, class_num):
    p_i = q_i.sum(0).view(-1)
    p_i /= p_i.sum()
    ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
    p_j = q_j.sum(0).view(-1)
    p_j /= p_j.sum()
    ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
    entropy = ne_i + ne_j

    q_i = q_i.t()
    q_j = q_j.t()
    N = 2 * class_num
    q = torch.cat((q_i, q_j), dim=0)
    temperature_l = 1
    similarity = nn.CosineSimilarity(dim=2)
    sim = similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l
    sim_i_j = torch.diag(sim, class_num)
    sim_j_i = torch.diag(sim, -class_num)

    def mask_correlated_samples(N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N)
    negative_clusters = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_clusters.device).long()
    logits = torch.cat((positive_clusters, negative_clusters), dim=1)
    loss = self.criterion(logits, labels)
    loss /= N
    return loss + entropy

def new_P( inputs, centers):
    alpha = 1
    q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(inputs, axis=1) - centers), axis=2) / alpha))
    q **= alpha
    q = np.transpose(np.transpose(q) / np.sum(q, axis=1))
    return q

def make_pseudo_label(model, features, A, class_num, view, device =None):
    model.eval()
    scaler = MinMaxScaler()
    z_list, s_list, z_fusion, q_list = model(features, A)
    for v in range(view):
        z_list[v] = z_list[v].cpu().detach().numpy()
        z_list[v] = scaler.fit_transform(z_list[v])
    outputall = np.hstack(z_list)
    km = KMeans(n_clusters=class_num, n_init=100)
    y_pred = km.fit_predict(outputall)
    Center_init = km.cluster_centers_
    Q = new_P(outputall, Center_init)
    return Q


class Graph_MLP_model(nn.Module):
    def __init__(self, n_in , view_num, cfg = None, dropout = 0.2,sparse = True, nb_classes = 5):
        super(Graph_MLP_model, self).__init__()
        self.view_num = view_num
        MLP = local_make_mlplayers(n_in, cfg, batch_norm=False)
        self.MLP_list = get_clones(MLP, self.view_num)
        self.nb_classes = nb_classes
        self.fc = nn.Linear(cfg[-1], self.nb_classes)
        self.dropout = dropout
        self.A = None
        self.sparse = sparse
        self.cfg = cfg

        # self.label_contrastive_module = nn.Sequential(
        #     # nn.Linear(cfg[-1], cfg[-1]),
        #     # act_layer('gelu'),
        #     nn.Linear(cfg[-1], self.nb_classes),
        #     # nn.Softmax(dim=1)
        # )
        self.label_contrastive_module = []
        self.transformer_module = []
        for v in range(view_num):
            self.transformer_module.append(DIFFormerConv(cfg[-1], cfg[-1], num_heads=3))
            self.label_contrastive_module.append(nn.Linear(cfg[-1], self.nb_classes))
        self.transformer_module = nn.ModuleList(self.transformer_module)
        self.label_contrastive_module = nn.ModuleList(self.label_contrastive_module)

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

        view_num = self.view_num

        x_list = [F.dropout(x, self.dropout, training=self.training) for i in range(view_num)]

        z_list = [self.MLP_list[i](x_list[i]) for i in range(view_num)]
        # q_list = [self.label_contrastive_module(z_list[i]) for i in range(view_num)]
        alpha = 0.5
        z_list = [self.transformer_module[i](z_list[i], z_list[i]) * alpha + z_list[i] * (1 - alpha) for i in range(view_num)]
        q_list = [self.label_contrastive_module[i](z_list[i]) for i in range(view_num)]
        s_list = [get_feature_dis(z_list[i]) for i in range(view_num)]

        # simple average
        z_unsqu = [z.unsqueeze(0) for z in z_list]
        z_fusion = torch.mean(torch.cat(z_unsqu), 0)

        return z_list, s_list, z_fusion, q_list

    def embed(self,  x , adj_list=None ):
        view_num = len(adj_list)

        x_list = [x for i in range(view_num)]

        z_list = [self.MLP_list[i](x_list[i]) for i in range(view_num)]

        # s_list = [get_feature_dis(z_list[i]) for i in range(view_num)]

        # simple average
        z_unsqu = [z.unsqueeze(0) for z in z_list]
        z_fusion = torch.mean(torch.cat(z_unsqu), 0)


        return z_fusion.detach()
    # def forward(self, x, adj_list=None):
    #     if self.A is None:
    #         self.A = adj_list[0]
    #
    #     x = F.dropout(x, self.dropout, training=self.training)
    #
    #     z = self.MLP(x)
    #     p = self.fc(z)
    #
    #     return z, p
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

def KDloss_0(input, target):
    T = 1  #todo
    target = F.softmax(target / T, dim=1)
    input = F.softmax(input / T, dim=1)
    l_kl = F.kl_div(input, target, size_average=False, log_target = False) * (T ** 2) / target.shape[0]
    return l_kl

class Graph_MLP(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.cfg = args.cfg
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):

        features = self.features.to(self.args.device)
        try:
            nb_classes = self.labels.shape[1]
        except:
            nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        view_num = 2
        A = self.adj_list[0].to(self.args.device)
        adj_label_list = [get_A_r(A, i) for i in range(view_num)]

        N = features.size(0)
        pri_Q = []
        I_target = torch.tensor(np.eye(nb_classes)).to(self.args.device)
        # print("Started training...")self.cfg[-1]

        model = Graph_MLP_model(self.args.ft_size, view_num, cfg=self.cfg, dropout=self.args.dropout, nb_classes = nb_classes).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr)

        model.train()
        start = time.time()

        # for epoch in tqdm(range(self.args.nb_epochs)):
        y_fake = None
        for epoch in range(self.args.nb_epochs):
            model.train()
            optimiser.zero_grad()
            z_list, s_list, z_fusion, q_list = model(features, A)
            # if y_fake is None:
            #     softmax_func = nn.Softmax(dim=1)
            #     y_fake = softmax_func(p.detach()/0.01)
            #     y_fake = torch.mm(A, y_fake)
            #     y_fake = softmax_func(y_fake/0.002)
            # s = get_feature_dis(z)
            loss = 0
            for i in range(view_num):
                loss_local = local_preserve(s_list[i], adj_label_list[i], tau=self.args.tau)
                loss_C, loss_simi = CCASSL(q_list, N, I_target, view_num)
                loss_p = 0
                if y_fake is not None:
                    loss_p = KDloss_0(q_list[i], y_fake)*1
                loss += (1 - loss_simi + loss_C * self.args.w_c)*0.1 + loss_local * self.args.w_l
                loss += loss_local + loss_p
            loss.backward()
            optimiser.step()

            if epoch % 100 == 0 and epoch > 0:
                model.eval()

                Q = make_pseudo_label(model, features, A, nb_classes, view_num)
                Q = target_distribution(Q, 1.5)
                pri_Q.append(Q)
                y_fake = torch.from_numpy(Q).to(self.args.device)
                z_list, s_list, z_fusion, q_list = model(features, A)

                # softmax_func = nn.Softmax(dim=1)
                # y_fake = softmax_func(p.detach()/0.01)
                # y_fake = torch.mm(A, y_fake)
                # y_fake = softmax_func(y_fake/0.002)

                acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, k1, k2, st = evaluate(
                    z_fusion.detach(), self.idx_train, self.idx_val, self.idx_test, self.labels,
                    seed=self.args.seed, epoch=self.args.test_epo, lr=self.args.test_lr)
                print(acc)

        training_time = time.time() - start
        # print("training time:{}s".format(training_time))
        # print("Evaluating...")
        model.eval()
        hf = model.embed(features, A)
        acc, acc_std, macro_f1,macro_f1_std, micro_f1,micro_f1_std,k1, k2, st = evaluate(
            hf, self.idx_train, self.idx_val, self.idx_test, self.labels,
            seed=self.args.seed, epoch=self.args.test_epo,lr=self.args.test_lr)
        return acc, acc_std, macro_f1,macro_f1_std, micro_f1,micro_f1_std,k1, k2, st,training_time

