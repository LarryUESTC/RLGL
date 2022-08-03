import time
from utils import process
from models.embedder import embedder_single
import os
from evaluate import evaluate, accuracy
from models.Layers import act_layer
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn


def get_A_r(adj, r):
    adj_label = adj
    for i in r - 1:
        adj_label = adj_label @ adj

    return adj_label

def Ncontrast(x_dis, adj_label, tau = 1):
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss


class MLP_Model(nn.Module):
    def __init__(self, n_in ,cfg = None, batch_norm=False, act='relu', dropout = 0.0, final_mlp = 0):
        super(MLP_Model, self).__init__()

        self.dropout = dropout
        self.bat = batch_norm
        self.act = act
        self.final_mlp = final_mlp > 0
        self.cfg = cfg
        self.layer_num = len(cfg)
        MLP_layers = []
        act_layers = []
        bat_layers = []
        in_channels = n_in
        for i, v in enumerate(cfg):
            out_channels = v
            MLP_layers.append(nn.Linear(in_channels, out_channels))
            if act:
                act_layers.append(act_layer(act))
            if batch_norm:
                bat_layers.append(nn.BatchNorm1d(out_channels, affine=False))
            in_channels = out_channels

        self.MLP_layers = nn.Sequential(*MLP_layers)
        self.act_layers = nn.Sequential(*act_layers)
        self.bat_layers = nn.Sequential(*bat_layers)
        # self.dop_layers = nn.Sequential(*dop_layers)
        if self.final_mlp:
            self.mlp = nn.Linear(in_channels, final_mlp)
        else:
            self.mlp = None

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        if isinstance(m, nn.Sequential):
            for mm in m:
                try:
                    torch.nn.init.xavier_uniform_(mm.weight.data)
                except:
                    pass


    def get_embedding(self, X_a):
        for i in range(self.layer_num):
            X_a = self.MLP_layers[i](X_a)
            if self.act:
                X_a = self.act_layers[i](X_a)
            if self.bat:
                X_a = self.bat_layers[i](X_a)

        if self.final_mlp:
            embeding_a = self.mlp(X_a)
        else:
            embeding_a = X_a
        return embeding_a.detach()


    def forward(self,  X_a):
        # X_a = F.dropout(X_a, 0.2)
        X_a = F.dropout(X_a, self.dropout, training=self.training)
        for i in range(self.layer_num):
            X_a = self.MLP_layers[i](X_a)
            if self.act and i != self.layer_num-1:
                X_a = self.act_layers[i](X_a)
            if self.bat and i != self.layer_num-1:
                X_a = self.bat_layers[i](X_a)
            if self.dropout > 0 and i != self.layer_num-1:
                X_a = F.dropout(X_a, self.dropout, training=self.training)
        if self.final_mlp:
            embeding_a = self.mlp(X_a)
        else:
            embeding_a = X_a

        return embeding_a


class GCNMLP(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.cfg = args.cfg

        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.cfg.append(nb_classes)
        self.model = MLP_Model(self.args.ft_size, cfg=self.cfg, final_mlp = 0,  dropout=self.args.random_aug_feature).to(self.args.device)

    def training(self):

        features = self.features.to(self.args.device)
        graph_org = self.dgl_graph.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)
        print("Started training...")


        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay = self.args.wd)
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
        for epoch in range(self.args.nb_epochs):
            self.model.train()
            optimiser.zero_grad()

            embeds = self.model(features)
            embeds_preds = torch.argmax(embeds, dim=1)
            
            train_embs = embeds[self.idx_train]
            val_embs = embeds[self.idx_val]
            test_embs = embeds[self.idx_test]

            
            loss = F.cross_entropy(train_embs, train_lbls)

            loss.backward()
            totalL.append(loss.item())
            optimiser.step()

            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0 :
                self.model.eval()
                # A_a, X_a = process.RA(graph_org.cpu(), features, 0, 0)
                # A_a = A_a.add_self_loop().to(self.args.device)
                embeds = self.model(features)
                val_acc = accuracy(embeds[self.idx_val], val_lbls)
                test_acc = accuracy(embeds[self.idx_test], test_lbls)
                # print(test_acc.item())
                # early stop
                stop_epoch = epoch
                if val_acc > best:
                    best = val_acc
                    output_acc = test_acc.item()
                    cnt_wait = 0
                    # torch.save(model.state_dict(), 'saved_model/best_{}.pkl'.format(self.args.dataset))
                else:
                    cnt_wait += 1
                if cnt_wait == self.args.patience:
                    # print("Early stopped!")
                    break
            ################END|Eval|###############


        training_time = time.time() - start
        # print("training time:{}s".format(training_time))
        # print("Evaluating...")
        print("\t[Classification] ACC: {:.4f} | stop_epoch: {:}| training_time: {:.4f} ".format(
            output_acc, stop_epoch, training_time))
        return output_acc, training_time, stop_epoch