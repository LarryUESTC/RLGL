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
from tensorboardX import SummaryWriter
import setproctitle
setproctitle.setproctitle('PLLL')
from datetime import datetime
import wandb

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
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

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

class RecordeBuffer:
    def __init__(self):
        self.psudo_labels = []
        self.entropy = []
        self.lenth = 0
    def clear(self):
        del self.psudo_labels[:]
        del self.entropy[:]
    def update(self):
        self.lenth = len(self.psudo_labels)
        if self.lenth > 100:
            del self.psudo_labels[0]
            del self.entropy[0]

def calc_entropy(input_tensor):
    lsm = nn.LogSoftmax()
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(dim=-1)
    return entropy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def attention(q, k, v, d_k, mask=None, dropout=None, out_att = False):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # scores = torch.where(scores > 1.1 * scores.mean(), scores, torch.zeros_like(scores))

    if mask is not None:
        mask = mask.unsqueeze(0)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    # scores = torch.where(scores > scores.mean(dim = -1).unsqueeze(dim = -1 ), scores, torch.zeros_like(scores))
    # scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    if out_att:
        return output, scores
    else:
        return output, None

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

    def forward(self, q, k, v, mask=None, out_att = True):
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
        scores, out_A = attention(q, k, v, self.d_k, mask, self.dropout, out_att = out_att)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(0, 1).contiguous().view(bs, self.d_model)
        output = self.out(concat)

        return output, out_A

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
        gcn_conv_output.append(matmul(adj, x[:, i]) )  # [N, D]
    gcn_conv_output = torch.stack(gcn_conv_output, dim=1) # [N, H, D]
    return gcn_conv_output

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
        x1 = self.norm_1(x)
        x1, out_A = self.attn(x1, x1, x1, mask)
        x = x + self.dropout_1(x1)
        # x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x))
        return x, out_A

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

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=True):
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


class GAT_selfCon_Trans(nn.Module):
    def __init__(self, n_in ,cfg = None, batch_norm=True, act='leakyrelu', dropout = 0.3, final_mlp = 0, nb_classes = 7,nheads = 4, Trans_layer_num = 2):
        super(GAT_selfCon_Trans, self).__init__()
        self.dropout = dropout
        self.bat = batch_norm
        self.act = act
        self.final_mlp = final_mlp > 0
        self.cfg = cfg
        self.Trans_layer_num = Trans_layer_num
        self.nheads = nheads
        self.nclass = nb_classes #cfg[-1] if final_mlp == 0 else final_mlp
        self.hid_dim= cfg[-1] #if final_mlp == 0 else cfg[-1]
        self.layer_num = len(self.cfg) #- 1 if final_mlp == 0 else cfg[-1]
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
        # self.layers = get_clones(DIFFormerConv(self.hid_dim, self.hid_dim, self.nheads), self.Trans_layer_num)
        self.norm_trans = Norm(int(self.hid_dim / self.nheads))
        # self.cls = nn.Linear(self.hid_dim, self.nclass)
        # cfg_head = [self.hid_dim, 64, 32, self.nclass]
        # MLP_layers_head = []
        # act_layers_head = []
        # bat_layers_head = []
        # norm_layers_head = []
        # in_channels_head = n_in
        # norm_layers_head.append(Norm(in_channels))
        # for i, v in enumerate(cfg_head):
        #     out_channels = v
        #     norm_layers_head.append(Norm(out_channels))
        #     MLP_layers_head.append(nn.Linear(in_channels, out_channels))
        #     if act:
        #         act_layers_head.append(act_layer(act))
        #     if batch_norm:
        #         bat_layers_head.append(nn.BatchNorm1d(out_channels, affine=False))
        #     in_channels = out_channels
        #
        # self.MLP_layers_head = nn.Sequential(*MLP_layers_head)
        # self.act_layers_head = nn.Sequential(*act_layers_head)
        # self.bat_layers_head = nn.Sequential(*bat_layers_head)
        # self.norm_layers_head = nn.Sequential(*norm_layers_head)

    def forward(self, x_input, adj=None):

        x = self.norm_layers[0](x_input)
        for i in range(len(self.MLP_layers)):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.MLP_layers[i](x)
            x = self.act_layers[i](x)
            x = self.norm_layers[i+1](x)

        x_dis = get_feature_dis(x)

        # x = x.detach()
        x = F.dropout(x, self.dropout, training=self.training)
        # x_dis = get_feature_dis(self.norm_layers[-2](x))
        out_A_list = []
        for i in range(self.Trans_layer_num):
            x, out_A = self.layers[i](x)
            out_A_list.append(out_A.detach())

        # x_dis = get_feature_dis(x)

        D_dim_single = int(self.hid_dim/self.nheads)
        CONN_INDEX = torch.zeros((x.shape[0],self.nclass)).to(x.device)
        for Head_i in range(self.nheads):
            feature_cls_sin = x[:, Head_i*D_dim_single:(Head_i+1)*D_dim_single]
            feature_cls_sin = self.norm_trans(feature_cls_sin)
            Linear_out_one = self.Linear_selfC[Head_i](feature_cls_sin)
            # CONN_INDEX += F.softmax(Linear_out_one - Linear_out_one.sort(descending= True)[0][:,3].unsqueeze(1), dim=1)
            CONN_INDEX += F.softmax(Linear_out_one, dim=1)
        # x_dis = get_feature_dis(CONN_INDEX)
        return F.log_softmax(CONN_INDEX, dim=1), x_dis, out_A_list


class SELFCONS(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.cfg = args.cfg

        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        # self.cfg.append(nb_classes)
        self.model = GAT_selfCon_Trans(self.args.ft_size, cfg=self.cfg, final_mlp = 0, nb_classes = nb_classes,
                                       dropout=self.args.random_aug_feature, nheads=self.args.nheads,
                                       Trans_layer_num =self.args.Trans_layer_num).to(self.args.device)

        current_time = datetime.now().strftime('%b%d_%H-%M:%S')
        self.TABLE_NAME = self.args.method + '_' + self.args.dataset
        logdir = os.path.join('runs5', current_time + '_selfcons_'+ str(self.args.seed))
        self.buffer = RecordeBuffer()
        self.writer_tb = SummaryWriter(log_dir = logdir)
        self.entropy_top = 500
        self.entropy = None
        self.psudo_labels = None
        self.top_entropy_idx = None
        self.N = None
        self.G = None
        self.top_k = 50

        self.run = wandb.init(
            # set the wandb project where this run will be logged
            project=self.TABLE_NAME,
            name = "exp-7-7-seed-" + str(self.args.seed),
            config=vars(self.args)
        )

    # def generate_G(self):
    #     pre_embedding_entropy = self.buffer.entropy[-1]
    #     pre_embedding = self.buffer.psudo_labels[-1]
    #     top_entropy_idx = torch.topk(pre_embedding_entropy, k=self.entropy_top, dim=-1, largest=False)[1]
    #     print(accuracy(pre_embedding[top_entropy_idx], self.labels[top_entropy_idx]))
    #     idx_train = self.idx_train.cpu().numpy().tolist()
    #     self.top_entropy_idx = []
    #     for i in top_entropy_idx:
    #         if i not in idx_train:
    #             self.top_entropy_idx.append(i.item())
    #     self.entropy = pre_embedding_entropy
    #     self.psudo_labels = pre_embedding.max(1)[1].type_as(self.labels)
    #     self.entropy_graph = -torch.log(pre_embedding_entropy.view(self.N, 1).repeat(1, self.N) * pre_embedding_entropy.view(1, self.N).repeat(self.N, 1) + 0.0001)
    #     self.lable_matrix = (self.psudo_labels.view(self.N, 1).repeat(1, self.N) == self.psudo_labels.view(1,self.N).repeat(self.N, 1)) + 0.0
    #     self.S = get_feature_dis(pre_embedding)
    #     # self.lable_matrix -=  ((self.psudo_labels.view(self.N, 1).repeat(1, self.N) != self.psudo_labels.view(1,self.N).repeat(self.N, 1)) + 0.0)*0.5
    #     G = torch.eye(self.N).to(self.args.device)
    #     A= torch.exp(self.S) * self.lable_matrix
    #     G[self.top_entropy_idx] = A[self.top_entropy_idx]
    #     # G = self.lable_matrix * self.entropy_graph * 0.1
    #     self.G = G
    #     return G

    def training(self):

        features = self.features.to(self.args.device)
        graph_org = self.dgl_graph.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)
        self.N = graph_org_torch.size()[0]
        number_neibor_list = splite_nerbor(self.adj_list[-1])
        print("Started training...")


        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay = self.args.wd)
        xent = nn.CrossEntropyLoss()
        train_lbls = self.labels[self.idx_train]
        val_lbls = self.labels[self.idx_val]
        test_lbls = self.labels[self.idx_test]
        train_onehot = F.one_hot(train_lbls)
        p_sudo = []
        lable_adj = (self.labels.view(1, self.N) == self.labels.view(self.N, 1))+0.
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
        # for i in range(2,3):
        #     adj_label = get_A_r(graph_org_torch, i)
        #     adj_label_list.append(adj_label)
        #
        # adj_label = adj_label_list[0]
        adj_label = graph_org_torch @ graph_org_torch
        # adj_label = F.normalize((adj_label_list[-2] > 0) + 0.0, p =1 )
        change_head_E = []
        change_Layer_E = []
        max_min_attention_E = []
        max_value_E = []
        max_index_E = []
        min_value_E = []
        min_index_E = []
        A_acc_E = []
        A_attention_E = []
        A_orggraph_E = []

        for epoch in range(self.args.nb_epochs):
            self.model.train()
            optimiser.zero_grad()
            # if epoch == 200 and epoch!=0:
            #     self.entropy_top = 1000
            #     G = self.generate_G()
            #     G_A = G #F.normalize(G, p=1) #+ graph_org_torch
            #     adj_label = G_A
            input_feature = features
            embeds, x_dis, out_A_list = self.model(input_feature)
            self.buffer.psudo_labels.append(embeds.detach())
            self.buffer.entropy.append(calc_entropy(embeds).detach())
            # adj_g = get_feature_dis(embeds).detach()
            # N = adj_g.shape[0]
            # maxadj = torch.topk(adj_g, k=30, dim=1, sorted=False, largest=True).values[:, -1].view(N, 1).repeat(1, N)
            # adj_g = adj_g * ((adj_g >= maxadj) + 0)
            # adj_g = adj_g@graph_org_torch
            embeds_preds = torch.argmax(embeds, dim=1)
            train_embs = embeds[self.idx_train]
            val_embs = embeds[self.idx_val]
            test_embs = embeds[self.idx_test]

            loss_Ncontrast = self.args.beta * Ncontrast(x_dis, adj_label, tau=self.args.tau)
            loss_ce = F.cross_entropy(train_embs, train_lbls)
            if self.top_entropy_idx is not None:
                loss_psudo = 0.2* F.cross_entropy(embeds[self.top_entropy_idx], self.psudo_labels[self.top_entropy_idx])
            else:
                loss_psudo = 0
            loss =  loss_ce + loss_Ncontrast #+ loss_psudo
            # loss = loss_cls + loss_Ncontrast * self.args.beta
            self.writer_tb.add_scalar('loss_ce', loss_ce.item(), epoch)
            self.writer_tb.add_scalar('loss_Ncontrast', loss_Ncontrast.item(), epoch)

            loss.backward()
            totalL.append(loss.item())
            optimiser.step()

            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0 :
                self.model.eval()
                embeds, _, out_A_list= self.model(features)
                val_acc = accuracy(embeds[self.idx_val], val_lbls)
                test_acc = accuracy(embeds[self.idx_test], test_lbls)
                self.writer_tb.add_scalar('Test_acc', test_acc, epoch)
                self.writer_tb.add_scalar('val_acc', val_acc, epoch)
                print(test_acc.item())

                change_head_L = []
                change_Layer_L = []
                max_min_attention_L_L = []
                max_value_L_L = []
                max_index_L_L = []
                min_value_L_L = []
                min_index_L_L = []
                A_acc_L_L = []
                A_attention_L_L = []
                A_orggraph_L_L = []
                for i in range(len(out_A_list)):
                    max_min_attention_L = []
                    max_value_L = []
                    max_index_L = []
                    min_value_L = []
                    min_index_L = []
                    A_acc_L = []
                    A_attention_L = []
                    A_orggraph_L = []
                    for head in range(len(out_A_list[i])):
                        A = out_A_list[i][head]
                        self.run.log({"idx_train_attGet{}{}".format(i, head): A[:, self.idx_train].sum(0).mean()})
                        self.run.log({"idx_val_attGet{}{}".format(i, head): A[:, self.idx_val].sum(0).mean()})
                        self.run.log({"idx_test_attGet{}{}".format(i, head): A[:, self.idx_test].sum(0).mean()})
                        self.run.log({"idx_train_acc_L_L{}{}".format(i, head): (A[self.idx_train] * lable_adj[self.idx_train]).sum(dim = 1).mean()})
                        self.run.log({"idx_val_acc_L_L{}{}".format(i, head): (A[self.idx_val] * lable_adj[self.idx_val]).sum(dim = 1).mean()})
                        self.run.log({"idx_test_acc_L_L{}{}".format(i, head): (A[self.idx_test] * lable_adj[self.idx_test]).sum(dim = 1).mean()})
                        A_acc_L.append(((A * lable_adj).sum(dim = 1)).cpu().numpy())
                        A_orggraph_L.append(((A * adj_label).sum(dim = 1) / adj_label.sum(dim = 1)).cpu().numpy())
                        node_attention = A.sum(dim=0)
                        A_attention_L.append(node_attention.cpu().numpy())
                        max_edge_attention = A.max().cpu().numpy()
                        min_edge_attention = A.min().cpu().numpy()
                        max_node_attention = node_attention.max().cpu().numpy()
                        min_node_attention = node_attention.min().cpu().numpy()

                        max_top = torch.topk(A, k=self.top_k, dim=1, sorted=True, largest=True)
                        min_top = torch.topk(A, k=self.top_k, dim=1, sorted=True, largest=False)
                        self.run.log({"max_ACC_L_L{}{}_T50".format(i, head): ((max_top[0] * lable_adj.gather(1, max_top[1])).sum(1) / max_top[0].sum(1)).mean()})
                        self.run.log({"min_ACC_L_L{}{}_T50".format(i, head): ((min_top[0] * lable_adj.gather(1, min_top[1])).sum(1) / min_top[0].sum(1)).mean()})
                        self.run.log({"max_ACC_L_L{}{}_T10".format(i, head): ((max_top[0][:, :10] * lable_adj.gather(1, max_top[1][:, :10])).sum(1) / max_top[0][:, :10].sum(1)).mean()})
                        self.run.log({"min_ACC_L_L{}{}_T10".format(i, head): ((min_top[0][:, :10] * lable_adj.gather(1, min_top[1][:, :10])).sum(1) / min_top[0][:, :10].sum(1)).mean()})
                        self.run.log({"max_ACC_L_L{}{}_T5".format(i, head): ((max_top[0][:, :5] * lable_adj.gather(1, max_top[1][:, :5])).sum(1) / max_top[0][:, :5].sum(1)).mean()})
                        self.run.log({"min_ACC_L_L{}{}_T5".format(i, head): ((min_top[0][:, :5] * lable_adj.gather(1, min_top[1][:, :5])).sum(1) / min_top[0][:, :5].sum(1)).mean()})
                        max_value_L.append(max_top[0].cpu().numpy())
                        max_index_L.append(max_top[1].cpu().numpy())
                        min_value_L.append(min_top[0].cpu().numpy())
                        min_index_L.append(min_top[1].cpu().numpy())
                        max_min_attention_L.append([max_edge_attention, min_edge_attention, max_node_attention, min_node_attention])

                    A_acc_L_L.append(A_acc_L)
                    A_attention_L_L.append(A_attention_L)
                    A_orggraph_L_L.append(A_orggraph_L)
                    max_value_L_L.append(max_value_L)
                    max_index_L_L.append(max_index_L)
                    min_value_L_L.append(min_value_L)
                    min_index_L_L.append(min_index_L)
                    max_min_attention_L_L.append(max_min_attention_L)
                    change_head = (out_A_list[i][1] - out_A_list[i][0]).abs().mean().cpu().numpy()
                    change_head_L.append(change_head)
                    if i >= 1:
                        change_Layer = (out_A_list[i][0] - out_A_list[i-1][0]).abs().mean().cpu().numpy()
                        change_Layer_L.append(change_Layer)

                change_head_E.append(change_head_L)
                change_Layer_E.append(change_Layer_L)
                max_min_attention_E.append(max_min_attention_L_L)
                max_value_E.append(max_value_L_L)
                max_index_E.append(max_index_L_L)
                min_value_E.append(min_value_L_L)
                min_index_E.append(min_index_L_L)
                A_acc_E.append(A_acc_L_L)
                A_attention_E.append(A_attention_L_L)
                A_orggraph_E.append(A_orggraph_L_L)

                self.run.log({"Epoch": epoch,
                              "test_acc": test_acc,
                              "val_acc": val_acc,
                             # "change_head_L": change_head_L,
                             #  "change_Layer_L": change_Layer_L,
                             #  "A_acc_L_L00": A_acc_L_L[0][0].mean(),
                             #  "A_acc_L_L01": A_acc_L_L[0][1].mean(),
                             #  "A_acc_L_L10": A_acc_L_L[1][0].mean(),
                             #  "A_acc_L_L11": A_acc_L_L[1][1].mean(),
                             #  "A_attention_L_L00": A_attention_L_L[0][0].max()-A_attention_L_L[0][0].min(),
                             #  "A_attention_L_L01": A_attention_L_L[0][1].max()-A_attention_L_L[0][1].min(),
                             #  "A_attention_L_L10": A_attention_L_L[1][0].max()-A_attention_L_L[1][0].min(),
                             #  "A_attention_L_L11": A_attention_L_L[1][1].max()-A_attention_L_L[1][1].min(),
                             #  "A_orggraph_L_L00": A_orggraph_L_L[0][0].mean(),
                             #  "A_orggraph_L_L01": A_orggraph_L_L[0][1].mean(),
                             #  "A_orggraph_L_L10": A_orggraph_L_L[1][0].mean(),
                             #  "A_orggraph_L_L11": A_orggraph_L_L[1][1].mean(),
                             #  "max_value_L_L00_T50": max_value_L_L[0][0].sum(1).mean(),
                             #  "min_value_L_L00_T50": min_value_L_L[0][0].sum(1).mean(),
                             #  "max_value_L_L10_T50": max_value_L_L[1][0].sum(1).mean(),
                             #  "min_value_L_L10_T50": min_value_L_L[1][0].sum(1).mean(),
                             #  "max_value_L_L00_T10": max_value_L_L[0][0][:, :10].sum(1).mean(),
                             #  "min_value_L_L00_T10": min_value_L_L[0][0][:, :10].sum(1).mean(),
                             #  "max_value_L_L10_T10": max_value_L_L[1][0][:, :10].sum(1).mean(),
                             #  "min_value_L_L10_T10": min_value_L_L[1][0][:, :10].sum(1).mean(),
                             #  "max_value_L_L00_T5": max_value_L_L[0][0][:, :5].sum(1).mean(),
                             #  "min_value_L_L00_T5": min_value_L_L[0][0][:, :5].sum(1).mean(),
                             #  "max_value_L_L10_T5": max_value_L_L[1][0][:, :5].sum(1).mean(),
                             #  "min_value_L_L10_T5": min_value_L_L[1][0][:, :5].sum(1).mean(),
                             #  "max_min_attention_L_L": max_min_attention_L_L,
                              })

                for i in range(len(change_head_L)):
                    self.run.log({"change_head_L_{}".format(i): change_head_L[i]})
                for i in range(len(change_Layer_L)):
                    self.run.log({"change_Layer_L_{}".format(i): change_Layer_L[i]})
                for l in range(len(A_acc_L_L)):
                    for h in range(len(A_acc_L_L[l])):
                        self.run.log({"A_acc_L_L{}{}".format(l,h): A_acc_L_L[l][h].mean()})
                for l in range(len(A_attention_L_L)):
                    for h in range(len(A_attention_L_L[l])):
                        self.run.log({"A_attention_L_L{}{}".format(l,h): A_attention_L_L[l][h].max() - A_attention_L_L[l][h].min()})
                for l in range(len(A_orggraph_L_L)):
                    for h in range(len(A_orggraph_L_L[l])):
                        self.run.log({"A_orggraph_L_L{}{}".format(l,h): A_orggraph_L_L[l][h].mean()})
                for l in range(len(max_value_L_L)):
                    for h in range(len(max_value_L_L[l])):
                        self.run.log({"max_value_L_L{}{}_T50".format(l, h): max_value_L_L[l][h].sum(1).mean()})
                        self.run.log({"min_value_L_L{}{}_T50".format(l, h): min_value_L_L[l][h].sum(1).mean()})
                        self.run.log({"max_value_L_L{}{}_T10".format(l, h): max_value_L_L[l][h][:, :10].sum(1).mean()})
                        self.run.log({"min_value_L_L{}{}_T10".format(l, h): min_value_L_L[l][h][:, :10].sum(1).mean()})
                        self.run.log({"max_value_L_L{}{}_T5".format(l, h): max_value_L_L[l][h][:, :5].sum(1).mean()})
                        self.run.log({"min_value_L_L{}{}_T5".format(l, h): min_value_L_L[l][h][:, :5].sum(1).mean()})
                # for test_ind in number_neibor_list:
                #     test_acc_ind = accuracy(embeds[test_ind], self.labels[test_ind])
                #     print("{:.2f} | ".format(test_acc_ind*100), end="")
                # print(test_acc.item())
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

        # str_now = datetime.now().strftime('%m-%d-%H-%M-%S')
        # filename = '{}.npz'.format(str_now)
        # npz_dir = "exp-7-5-seed-" + str(self.args.seed)
        # file_path = os.path.join('Temp',self.TABLE_NAME, npz_dir)
        # try:
        #     os.makedirs(file_path)
        # except:
        #     pass
        # np.savez(os.path.join(file_path,filename),
        #          change_head_E = np.array(change_head_E),
        #          change_Layer_E = np.array(change_Layer_E),
        #          max_min_attention_E=np.array(max_min_attention_E),
        #          max_value_E=np.array(max_value_E),
        #          max_index_E=np.array(max_index_E),
        #          min_value_E=np.array(min_value_E),
        #          min_index_E=np.array(min_index_E),
        #          A_acc_E=np.array(A_acc_E),
        #          A_attention_E=np.array(A_attention_E),
        #          A_orggraph_E=np.array(A_orggraph_E),
        #          )
        #
        # data_filename = os.path.join('Temp',self.TABLE_NAME, '{}.npz'.format(self.args.dataset))
        # if os.path.exists(data_filename):
        #     pass
        # else:
        #     np.savez(data_filename, adj=self.adj_list[-1].cpu().numpy(), label = self.labels.cpu().numpy(), idx_train = self.idx_train.cpu().numpy(), idx_val = self.idx_val.cpu().numpy(), idx_test = self.idx_test.cpu().numpy())
        #
        wandb.finish()
        return output_acc, training_time, stop_epoch

