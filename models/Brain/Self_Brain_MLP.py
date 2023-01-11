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
import random

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

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

class Norm2(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-2, keepdim=True)) \
               / (x.std(dim=-2, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) #/ math.sqrt(d_k)
    # scores = torch.where(scores > 1.1 * scores.mean(), scores, torch.zeros_like(scores))

    if mask is not None:
        mask = mask.unsqueeze(0)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    scores = torch.where(scores > scores.mean(dim = -1).unsqueeze(dim = -1 ), scores, torch.zeros_like(scores))
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
        self.out = nn.Linear(d_model_out, d_model_in)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        T = q.size(1)
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, T, self.h, self.d_k)
        q = self.q_linear(q).view(bs, T, self.h, self.d_k)
        v = self.v_linear(v).view(bs, T, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(2, 1)
        q = q.transpose(2, 1)
        v = v.transpose(2, 1)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, T, self.d_model)
        output = self.out(concat)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=128, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff//4)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff//4, d_ff)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm2(d_model)
        self.norm_2 = Norm2(d_model)
        self.attn = MultiHeadAttention_new(heads, d_model, d_model*heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask =None):
        x2 = self.norm_1(x)
        x2 = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        # x2 = self.norm_2(x)
        x = self.dropout_2(self.ff(x2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

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
        self.nheads = 8
        self.Translayers = nn.ModuleList([EncoderLayer(1, self.nheads, 8, self.dropout),
                                          EncoderLayer(8, self.nheads, 16, self.dropout)])

    def forward(self, x_input, dropout = 0.0):

        N, B, T = x_input.size()
        # x = x_input.view(N*B, T)
        # x = self.norm_layers[0](x)
        # for i in range(len(self.MLP_layers)):
        #     x = F.dropout(x, dropout, training=self.training)
        #     x = self.MLP_layers[i](x)
        #     x = self.act_layers[i](x)
        #     x = self.norm_layers[i+1](x)
        # x = self.Linear_selfC(x)

        x = x_input.view(N * B, T, 1)
        for i in range(self.Trans_layer_num):
            x = self.Translayers[i](x)
        x = x.mean(dim=-2)
        return x.view(N, B, -1)

def get_x(features_time, lenth):
    N, B, total_lenth = features_time.size()
    stat_point = random.randint(0, total_lenth - lenth - 1)
    out_time = features_time[:,:,stat_point:stat_point+lenth]
    return out_time

def get_xx(features_time, lenth, lenth_2):
    N, B, total_lenth = features_time.size()
    stat_point = min(random.randint(0, total_lenth - lenth - 1), total_lenth - lenth_2 - 1)
    out_time1 = features_time[:, :, stat_point:stat_point + lenth]
    out_time2 = features_time[:, :, stat_point:stat_point + lenth_2]
    return out_time1, out_time2

def get_x_test(features_time, lenth):
    N, B, total_lenth = features_time.size()
    window_size = int(total_lenth / lenth)
    out_time_list = []
    for i in range(window_size):
        out_time = features_time[:,:,lenth*i:lenth*(i+1)]
        out_time_list.append(out_time)
    return out_time_list


class SELFBRAINMLP(embedder_brain):
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
        self.batchsize = self.features_time.size()[0]

    def fold_train(self):
        current_time = datetime.now().strftime('%b%d_%H-%M:%S')
        logdir = os.path.join('runs7', current_time + '_selfbrain_'+ str(self.args.seed)+ '_flod_'+str(self.fold_num))
        self.writer_tb = SummaryWriter(log_dir = logdir)
        self.features = self.features_pearson[:,self.mask]
        self.N = self.features_pearson.size()[0]

        accs_ori, precision_ori, recall_ori, f1_ori, auc_ori = self.original_test()
        string_1 = Fore.GREEN + "Ori |accs: {:.3f},auc: {:.3f},pre: {:.3f},recall: {:.3f},f1: {:.3f}".format(accs_ori, auc_ori, precision_ori, recall_ori, f1_ori)
        print(string_1)

        self.cfg = [64,  64, 64, 32, 16]
        self.lenth = 50
        self.model = SSL_Trans(n_in = self.lenth, cfg= self.cfg)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        self.model = self.model.to(self.args.device)
        I_target = torch.tensor(np.eye(self.cfg[-1])).to(self.args.device)
        N_target = torch.tensor(np.eye(self.N)).to(self.args.device)
        B_target = torch.tensor(np.eye(200)).to(self.args.device) == 1
        batch_list = [i for i in range(self.train_index.size(0))]
        for epoch in range(self.args.nb_epochs):
            random.shuffle(batch_list)
            batch = batch_list[:8]
            self.model.train()
            optimiser.zero_grad()
            X_a, X_b = get_xx(self.features_time[self.train_index][batch], random.randint(20, 60), random.randint(20, 60))
            # X_b = get_x(self.features_time[self.train_index][batch], self.lenth)
            embeding_a = self.model(X_a)
            embeding_b = self.model(X_b)
            S = torch.bmm(embeding_a, torch.transpose(embeding_b, 2, 1))
            S_exp = torch.exp(S)
            pos = S_exp[:,B_target]
            dis = torch.sum(S_exp, -1)/200
            loss = -torch.log(pos * (dis ** (-1)) + 1e-8).mean()


            loss.backward()
            optimiser.step()
            # string_1 = Fore.GREEN + "Epoch:{:} |loss: {:.3f}".format(epoch, loss.item())
            # print(string_1)
            self.writer_tb.add_scalar('loss', loss.item(), epoch)

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
        # embeds = self.model(self.features).detach()
        # train_embs = embeds[self.train_index]
        # test_embs = embeds[self.test_index]
        # val_embs = embeds[self.val_index]
        train_labels = self.labels[self.train_index]
        test_labels = self.labels[self.test_index]
        val_labels = self.labels[self.val_index]

        #todo
        X_a_list = get_x_test(self.features_time, self.lenth)
        embeds = 0

        for X_a in X_a_list:
            embeds_list = []
            for i in range(X_a.size()[0]):
                embeding_a = self.model(X_a[i].unsqueeze(dim = 0)).detach()
                S = torch.bmm(embeding_a, torch.transpose(embeding_a, 2, 1))
                S_exp = torch.exp(S)
                embeds_list.append(S_exp[:, self.mask])
            embeds += torch.stack(embeds_list).squeeze()
        ''' Linear Evaluation '''
        train_embs = embeds[self.train_index]
        test_embs = embeds[self.test_index]
        val_embs = embeds[self.val_index]

        logreg = LogReg(train_embs.shape[1], 2)
        opt = torch.optim.Adam(logreg.parameters(), lr=0.01, weight_decay=0e-5)

        logreg = logreg.to(self.args.device)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch2 in range(200):
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

    def original_test(self):
        train_labels = self.labels[self.train_index]
        test_labels = self.labels[self.test_index]
        val_labels = self.labels[self.val_index]

        embeds = self.features
        train_embs = embeds[self.train_index]
        test_embs = embeds[self.test_index]
        val_embs = embeds[self.val_index]

        logreg = LogReg(train_embs.shape[1], 2)
        opt = torch.optim.Adam(logreg.parameters(), lr=0.01, weight_decay=0e-5)

        logreg = logreg.to(self.args.device)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch2 in range(200):
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