import time
from utils import process
from models.embedder import embedder_brain
import os
import numpy as np
from evaluate import evaluate, accuracy
from colorama import init, Fore, Back, Style
from models.Layers import SUGRL_Fast, GNN_Model, act_layer
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, roc_auc_score
import math
import copy
from tensorboardX import SummaryWriter
import setproctitle
setproctitle.setproctitle('PLLL')
from datetime import datetime

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

def Ncontrast(x_dis, adj_label, tau = 1):
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

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

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # scores = torch.where(scores > 1.1 * scores.mean(), scores, torch.zeros_like(scores))

    if mask is not None:
        mask = mask.unsqueeze(0)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    scores = torch.where(scores > scores.mean(), scores, torch.zeros_like(scores))
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


class SSL_Trans(nn.Module):
    def __init__(self, n_in ,cfg = None, batch_norm=True, act='leakyrelu', dropout = 0.1, final_mlp = 0):
        super(SSL_Trans, self).__init__()
        self.dropout = dropout
        self.bat = batch_norm
        self.act = act
        self.final_mlp = final_mlp > 0
        self.cfg = cfg
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

        self.Linear_selfC = nn.Linear(in_channels, self.hid_dim)

        self.Trans_layer_num = 2
        self.nheads = 4
        self.nclass = 2
        self.Linear_selfC = get_clones(nn.Linear(int(self.hid_dim / self.nheads), self.nclass), self.nheads)
        self.Trans_layers = get_clones(EncoderLayer(self.hid_dim, self.nheads, self.dropout), self.Trans_layer_num)
        self.norm_trans = Norm(int(self.hid_dim / self.nheads))

        # self.Linear_selfC = get_clones(nn.Linear(int(self.hid_dim / self.nheads), self.nclass), self.nheads)
        # self.layers = get_clones(EncoderLayer(self.hid_dim, self.nheads, self.dropout), self.Trans_layer_num)
        # self.norm_trans = Norm(int(self.hid_dim / self.nheads))

    def forward(self, x_input, dropout = 0.0):
        # x = self.norm_layers[0](x_input)
        x = x_input
        for i in range(len(self.MLP_layers)):
            x = F.dropout(x, dropout, training=self.training)
            x = self.MLP_layers[i](x)
            x = self.act_layers[i](x)
            x = self.norm_layers[i+1](x)
        x = self.Linear_selfC(x)
        return x

    def forward_g(self, x_input, dropout = 0.0):

        x = self.forward(x_input)
        x = x.detach()
        for i in range(self.Trans_layer_num):
            x = self.Trans_layers[i](x)

        D_dim_single = int(self.hid_dim/self.nheads)
        CONN_INDEX = torch.zeros((x.shape[0],self.nclass)).to(x.device)
        for Head_i in range(self.nheads):
            feature_cls_sin = x[:, Head_i*D_dim_single:(Head_i+1)*D_dim_single]
            feature_cls_sin = self.norm_trans(feature_cls_sin)
            Linear_out_one = self.Linear_selfC[Head_i](feature_cls_sin)
            # CONN_INDEX += F.softmax(Linear_out_one - Linear_out_one.sort(descending= True)[0][:,3].unsqueeze(1), dim=1)
            CONN_INDEX += F.softmax(Linear_out_one, dim=1)

        return F.log_softmax(CONN_INDEX, dim=1)

class Trans(nn.Module):
    def __init__(self, n_in,  dropout = 0.1):
        super(Trans, self).__init__()
        self.dropout = dropout
        self.hid_dim= n_in #if final_mlp == 0 else cfg[-1]
        self.Trans_layer_num = 2
        self.nheads = 4
        self.nclass = 2
        self.Linear_selfC = get_clones(nn.Linear(int(self.hid_dim / self.nheads), self.nclass), self.nheads)
        self.Trans_layers = get_clones(EncoderLayer(self.hid_dim, self.nheads, self.dropout), self.Trans_layer_num)
        self.norm_trans = Norm(int(self.hid_dim / self.nheads))

    def forward(self, x_input, dropout = 0.0):

        x = x_input
        for i in range(self.Trans_layer_num):
            x = self.Trans_layers[i](x)

        D_dim_single = int(self.hid_dim/self.nheads)
        CONN_INDEX = torch.zeros((x.shape[0],self.nclass)).to(x.device)
        for Head_i in range(self.nheads):
            feature_cls_sin = x[:, Head_i*D_dim_single:(Head_i+1)*D_dim_single]
            feature_cls_sin = self.norm_trans(feature_cls_sin)
            Linear_out_one = self.Linear_selfC[Head_i](feature_cls_sin)
            # CONN_INDEX += F.softmax(Linear_out_one - Linear_out_one.sort(descending= True)[0][:,3].unsqueeze(1), dim=1)
            CONN_INDEX += F.softmax(Linear_out_one, dim=1)

        return F.log_softmax(CONN_INDEX, dim=1)

class SELFBRAIN(embedder_brain):
    def __init__(self, args):
        embedder_brain.__init__(self, args)
        self.args = args
        # self.cfg = args.cfg
        # nb_classes = (self.labels.max() - self.labels.min() + 1).item()

        self.entropy = None
        self.psudo_labels = None
        self.top_entropy_idx = None
        self.N = None
        self.G = None
        self.mask = torch.triu((torch.ones(200,200) == 1),diagonal =1)
        self.input_dim = self.mask.sum().item()

    def fold_train(self):
        current_time = datetime.now().strftime('%b%d_%H-%M:%S')
        logdir = os.path.join('runs6', current_time + '_selfbrain_'+ str(self.args.seed)+ '_flod_'+str(self.fold_num))
        self.buffer = RecordeBuffer()
        self.writer_tb = SummaryWriter(log_dir = logdir)
        self.features = self.features_pearson[:,self.mask]
        self.N = self.features_pearson.size()[0]
        self.cfg = [512,  512, 128, 128]
        self.model = SSL_Trans(n_in = self.input_dim, cfg= self.cfg)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        self.model = self.model.to(self.args.device)
        I_target = torch.tensor(np.eye(self.cfg[-1])).to(self.args.device)
        N_target = torch.tensor(np.eye(self.N)).to(self.args.device)

        for epoch in range(self.args.nb_epochs):
            self.model.train()
            optimiser.zero_grad()
            X_a = F.dropout(self.features, 0.0)
            X_b = F.dropout(self.features, 0.8)

            embeding_a = self.model(X_a)
            embeding_b = self.model(X_b)

            embeding_a = (embeding_a - embeding_a.mean(0)) / embeding_a.std(0)
            embeding_b = (embeding_b - embeding_b.mean(0)) / embeding_b.std(0)

            c1 = torch.mm(embeding_a.T, embeding_a)/ self.N
            c2 = torch.mm(embeding_b.T, embeding_b)/ self.N

            # dis_a = get_feature_dis(embeding_a)
            # loss_Ncontrast = 0.1 * Ncontrast(dis_a, N_target)

            loss_c1 = (I_target - c1).pow(2).mean() + torch.diag(c1).mean()
            loss_c2 = (I_target - c2).pow(2).mean() + torch.diag(c2).mean()
            loss_C = loss_c1 + loss_c2
            loss_simi = cosine_similarity(embeding_a, embeding_b.detach(), dim=-1).mean()
            loss = 1 - loss_simi + loss_C*5

            loss.backward()
            optimiser.step()
            self.writer_tb.add_scalar('loss_C', loss_C.item(), epoch)
            self.writer_tb.add_scalar('loss_simi', loss_simi.item(), epoch)

            if epoch % 10 == 0 and epoch != 0:
                accs, precision, recall, f1, auc = self.flod_test()
                string_1 = Fore.GREEN + "Epoch:{:} |accs: {:.3f},auc: {:.3f},pre: {:.3f},recall: {:.3f},f1: {:.3f}".format(epoch, accs, auc, precision, recall, f1)
                print(string_1)
                self.writer_tb.add_scalar('accs', accs, epoch)
                self.writer_tb.add_scalar('precision', precision, epoch)
                self.writer_tb.add_scalar('recall', recall, epoch)
                self.writer_tb.add_scalar('f1', f1, epoch)
                self.writer_tb.add_scalar('auc', auc, epoch)


        output_acc = 0
        stop_epoch = 0

        return output_acc, stop_epoch

    def flod_test(self):
        self.model.eval()
        embeds = self.model(self.features).detach()
        train_embs = embeds[self.train_index]
        test_embs = embeds[self.test_index]
        val_embs = embeds[self.val_index]
        train_labels = self.labels[self.train_index]
        test_labels = self.labels[self.test_index]
        val_labels = self.labels[self.val_index]
        ''' Linear Evaluation '''
        logreg = LogReg(train_embs.shape[1], 2)
        opt = torch.optim.Adam(logreg.parameters(), lr=0.01, weight_decay=0e-5)

        logreg = logreg.to(self.args.device)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch2 in range(100):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            loss2 = loss_fn(logits, train_labels.long())
            loss2.backward()
            opt.step()

        logreg.eval()
        with torch.no_grad():
            test_logits = logreg(test_embs)
            pred = torch.argmax(test_logits, dim=1)

            accs = accuracy_score(test_labels.cpu(), pred.cpu())
            precision = precision_score(test_labels.cpu(), pred.cpu())
            recall = recall_score(test_labels.cpu(), pred.cpu())
            f1 = f1_score(test_labels.cpu(), pred.cpu())
            try:
                auc = roc_auc_score(test_labels.cpu(), pred.cpu())
            except:
                auc = f1 * 0

        return accs, precision, recall, f1, auc

    def training(self):
        acc_list_SSL = []
        auc_list_SSL = []
        precision_list_SSL = []
        recall_list_SSL = []
        f1_lis_SSLt = []
        acc_list = []
        auc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        self.fold_num = 0
        for train_index, test_index, val_list in zip(self.train_list, self.test_list, self.val_list):
            self.fold_num += 1
            self.train_index = train_index
            self.test_index = test_index
            self.val_index = val_list
            output_acc, stop_epoch = self.fold_train()
        return output_acc, stop_epoch