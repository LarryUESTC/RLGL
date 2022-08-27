import time
from utils import process
from models.embedder import embedder_single
import os
from evaluate import evaluate, accuracy
from torch.nn.modules.activation import MultiheadAttention
from models.Layers import SUGRL_Fast, GNN_Model, act_layer
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import copy

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

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).cuda(3)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

# def get_feature_dis_New(x):
#     """
#     x :           batch_size x nhid
#     x_dis(i,j):   item means the similarity between x(i) and x(j).
#     """
#     x_dis = x@x.T
#     N = x_dis.shape[0]
#     maxadj = torch.topk(x_dis, k=10, dim=1, sorted=False, largest=True).values[:, -1].view(N, 1).repeat(1, N)
#     x_dis = x_dis * ((x_dis >= maxadj) + 0)
#     mask = torch.eye(x_dis.shape[0]).cuda()
#     x_sum = torch.sum(x**2, 1).reshape(-1, 1)
#     x_sum = torch.sqrt(x_sum).reshape(-1, 1)
#     x_sum = x_sum @ x_sum.T
#     x_dis = x_dis*(x_sum**(-1))
#     x_dis = (1-mask) * x_dis
#
#     return x_dis

def splite_nerbor(adj, max_num = 10):
    number_neibor = adj.sum(dim=1)
    number_neibor_zero = torch.zeros_like(number_neibor)
    number_neibor_one = torch.ones_like(number_neibor)
    number_neibor_list = []
    for num in range(1, max_num):
        number_neibor_list.append(torch.nonzero(number_neibor == num).squeeze())
    number_neibor_list.append(
        torch.nonzero(torch.where(number_neibor >= max_num, number_neibor_one, number_neibor_zero)).squeeze())
    return number_neibor_list


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # scores = torch.where(scores > 1.1 * scores.mean(), scores, torch.zeros_like(scores))

    if mask is not None:
        mask = mask.unsqueeze(0)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

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
        # q = k.clone()
        # v = k.clone()

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

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention_new(heads, d_model,d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask =None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class GAT_selfCon_Trans(nn.Module):
    def __init__(self, n_in ,cfg = None, batch_norm=True, act='leakyrelu', dropout = 0.2, final_mlp = 0, nheads = 4, Trans_layer_num = 2):
        super(GAT_selfCon_Trans, self).__init__()
        self.dropout = dropout
        self.bat = batch_norm
        self.act = act
        self.final_mlp = final_mlp > 0
        self.cfg = cfg
        self.Trans_layer_num = Trans_layer_num
        self.nheads = nheads
        self.nclass = cfg[-1] if final_mlp == 0 else final_mlp
        self.hid_dim= cfg[-2] if final_mlp == 0 else cfg[-1]
        self.layer_num = len(self.cfg)
        MLP_layers = []
        act_layers = []
        bat_layers = []
        norm_layers = []
        in_channels = n_in
        norm_layers.append(Norm(in_channels))
        for i, v in enumerate(cfg):
            out_channels = v
            norm_layers.append(Norm(out_channels))
            MLP_layers.append(nn.Linear(in_channels, out_channels))
            if act:
                act_layers.append(act_layer(act))
            if batch_norm:
                bat_layers.append(nn.BatchNorm1d(out_channels, affine=False))
            in_channels = out_channels

        self.MLP_layers = nn.Sequential(*MLP_layers)
        self.act_layers = nn.Sequential(*act_layers)
        self.bat_layers = nn.Sequential(*bat_layers)
        self.norm_layers = nn.Sequential(*norm_layers)


        # self.pe = nn.Linear(self.nsample, self.nhid)

        self.Linear_selfC = get_clones(nn.Linear(int(self.hid_dim / self.nheads), self.nclass), self.nheads)
        self.layers = get_clones(EncoderLayer(self.hid_dim, self.nheads, self.dropout), self.Trans_layer_num)
        self.norm_trans = Norm(int(self.hid_dim / self.nheads))
        # self.cls = nn.Linear(self.hid_dim, self.nclass)


    def forward(self, x_input, adj=None):

        x_input = self.norm_layers[0](x_input)
        x = self.MLP_layers[0](x_input)
        x = F.dropout(x, self.dropout, training=self.training)

        x_dis = get_feature_dis(self.norm_layers[-2](x))

        for i in range(self.Trans_layer_num):
            x = self.layers[i](x)

        x_dis_1 = get_feature_dis(self.norm_layers[-2](x))
        D_dim_single = int(self.hid_dim/self.nheads)
        CONN_INDEX = torch.zeros((x.shape[0],self.nclass)).to(x.device)

        for Head_i in range(self.nheads):
            feature_cls_sin = x[:, Head_i*D_dim_single:(Head_i+1)*D_dim_single]
            feature_cls_sin = self.norm_trans(feature_cls_sin)
            Linear_out_one = F.elu(self.Linear_selfC[Head_i](feature_cls_sin))
            CONN_INDEX += F.softmax(Linear_out_one - Linear_out_one.sort(descending= True)[0][:,3].unsqueeze(1), dim=1)



        return F.log_softmax(CONN_INDEX, dim=1), x_dis, x_dis_1


class SELFCONS(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.cfg = args.cfg

        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.cfg.append(nb_classes)
        self.model = GAT_selfCon_Trans(self.args.ft_size, cfg=self.cfg, final_mlp = 0, dropout=self.args.random_aug_feature, nheads=self.args.nheads, Trans_layer_num =self.args.Trans_layer_num).to(self.args.device)

    def training(self):

        features = self.features.to(self.args.device)
        graph_org = self.dgl_graph.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)

        number_neibor_list = splite_nerbor(self.adj_list[-1])
        print("Started training...")


        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay = self.args.wd)
        xent = nn.CrossEntropyLoss()
        train_lbls = self.labels[self.idx_train]
        val_lbls = self.labels[self.idx_val]
        test_lbls = self.labels[self.idx_test]
        train_onehot = F.one_hot(train_lbls)
        p_sudo = []

        cnt_wait = 0
        best = 1e-9
        output_acc = 1e-9
        stop_epoch = 0

        start = time.time()
        totalL = []
        N = features.size(0)
        # features = F.normalize(features)
        targets_u = None
        adj_label_list = []
        for i in range(2,5):
            adj_label = get_A_r(graph_org_torch, i)
            adj_label_list.append(adj_label)

        adj_label = adj_label_list[-1]
        for epoch in range(self.args.nb_epochs):
            self.model.train()
            optimiser.zero_grad()


            input_feature = features
            embeds, x_dis, x_dis_1 = self.model(input_feature)

            embeds_preds = torch.argmax(embeds, dim=1)
            train_embs = embeds[self.idx_train]
            val_embs = embeds[self.idx_val]
            test_embs = embeds[self.idx_test]

            loss_Ncontrast = Ncontrast(x_dis, adj_label, tau=self.args.tau)
            loss_Ncontrast_1 = Ncontrast(x_dis_1, adj_label, tau=self.args.tau)

            loss_cls = F.cross_entropy(train_embs, train_lbls)
            loss = loss_cls + loss_Ncontrast * self.args.beta

            loss.backward()
            totalL.append(loss.item())
            optimiser.step()

            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0 :
                self.model.eval()
                embeds, _, _ = self.model(features)
                val_acc = accuracy(embeds[self.idx_val], val_lbls)
                test_acc = accuracy(embeds[self.idx_test], test_lbls)
                # print(test_acc.item())
                for test_ind in number_neibor_list:
                    test_acc_ind = accuracy(embeds[test_ind], self.labels[test_ind])
                    print("{:.2f} | ".format(test_acc_ind*100), end="")
                print(test_acc.item())
                # early stop
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
        return output_acc, training_time, stop_epoch