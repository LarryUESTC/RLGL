import time
from utils import process
from models.embedder import embedder_brain
import os
import numpy as np
from evaluate import evaluate, accuracy
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
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

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

class SSL_Trans(nn.Module):
    def __init__(self, n_in ,cfg = None, batch_norm=True, act='leakyrelu', dropout = 0.1, final_mlp = 0,nheads = 4, Trans_layer_num = 2):
        super(SSL_Trans, self).__init__()
        self.dropout = dropout
        self.bat = batch_norm
        self.act = act
        self.final_mlp = final_mlp > 0
        self.cfg = cfg
        self.Trans_layer_num = Trans_layer_num
        self.nheads = nheads
        self.nclass = 2 #cfg[-1] if final_mlp == 0 else final_mlp
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

        # self.Linear_selfC = get_clones(nn.Linear(int(self.hid_dim / self.nheads), self.nclass), self.nheads)
        # self.layers = get_clones(EncoderLayer(self.hid_dim, self.nheads, self.dropout), self.Trans_layer_num)
        # self.norm_trans = Norm(int(self.hid_dim / self.nheads))


    def forward(self, x_input, dropout = 0.1):

        x = self.norm_layers[0](x_input)
        for i in range(len(self.MLP_layers)):
            x = F.dropout(x, dropout, training=self.training)
            x = self.MLP_layers[i](x)
            x = self.act_layers[i](x)
            x = self.norm_layers[i+1](x)
        x = self.Linear_selfC(x)
        return x

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
        self.cfg = [1024, 512, 512, 512, 128, 128]
        self.model = SSL_Trans(n_in = self.input_dim, cfg= self.cfg)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        self.model = self.model.to(self.args.device)
        I_target = torch.tensor(np.eye(self.cfg[-1])).to(self.args.device)

        for epoch in range(self.args.nb_epochs):
            self.model.train()
            optimiser.zero_grad()
            X_a = F.dropout(self.features, 0.4)
            X_b = F.dropout(self.features, 0.2)

            embeding_a = self.model(X_a)
            embeding_b = self.model(X_b)

            embeding_a = (embeding_a - embeding_a.mean(0)) / embeding_a.std(0)
            embeding_b = (embeding_b - embeding_b.mean(0)) / embeding_b.std(0)

            c1 = torch.mm(embeding_a.T, embeding_a)/ self.N
            c2 = torch.mm(embeding_b.T, embeding_b)/ self.N


            loss_c1 = (I_target - c1).pow(2).mean() + torch.diag(c1).mean()
            loss_c2 = (I_target - c2).pow(2).mean() + torch.diag(c2).mean()
            loss_C = loss_c1 + loss_c2
            loss_simi = cosine_similarity(embeding_a, embeding_b.detach(), dim=-1).mean()
            loss = 1 - loss_simi + loss_C

            loss.backward()
            optimiser.step()

            if epoch % 50 == 0 and epoch != 0:
                accs, precision, recall, f1, auc = self.flod_test()
                print(accs)


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