import time
from utils import process
from models.embedder import embedder_single
import os
from evaluate import evaluate, accuracy
from models.Layers import GNN_layer, act_layer
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

def get_similarity(x):
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

def Ncontrast(x_dis, adj_label, tau = 1):
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(adj_label, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

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

class RGNN_Model(nn.Module):
    def __init__(self, n_in, cfg=None, batch_norm=False, act='relu', gnn='GCN', dropout=0.0, final_mlp=0):
        super(RGNN_Model, self).__init__()

        self.dropout = dropout
        self.bat = batch_norm
        self.act = act
        self.gnn = gnn
        self.final_mlp = final_mlp > 0
        self.A = None
        self.sparse = True
        self.cfg = cfg
        self.layer_num = len(cfg)
        GCN_layers = []
        act_layers = []
        bat_layers = []
        in_channels = n_in
        self.reverse_layer = nn.Linear(in_channels, cfg[0])
        self.P_layer = nn.Linear(cfg[-1], in_channels)
        for i, v in enumerate(cfg):
            out_channels = v
            GCN_layers.append(GNN_layer(gnn, in_channels, out_channels))
            if act:
                act_layers.append(act_layer(act))
            if batch_norm:
                bat_layers.append(nn.BatchNorm1d(out_channels, affine=False))
            in_channels = out_channels

        self.GCN_layers = nn.Sequential(*GCN_layers)
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

    def get_embedding(self, A_a, X_a):
        for i in range(self.layer_num):
            X_a = self.GCN_layers[i](A_a, X_a)
            if self.act:
                X_a = self.act_layers[i](X_a)
            if self.bat:
                X_a = self.bat_layers[i](X_a)

        if self.final_mlp:
            embeding_a = self.mlp(X_a)
        else:
            embeding_a = X_a
        return embeding_a.detach()

    def forward(self, A_a, X_a):
        # X_a = F.dropout(X_a, 0.2)
        # X_a = F.dropout(X_a, self.dropout, training=self.training)
        N = X_a.size(0)
        z = self.reverse_layer(X_a)
        S = get_similarity(z)
        topk = 5
        v = 0.5
        maxadj = torch.topk(S, k=topk, dim=1, sorted=False, largest=True).values[:, -1].view(N, 1).repeat(1, N)
        S_out = S * ((S >= maxadj) + 0)
        adj = (1 - v) * A_a + v * S_out
        for i in range(self.layer_num):
            X_a = self.GCN_layers[i](adj, X_a)
            if self.gnn == "GAT":
                X_a = torch.mean(X_a, dim=1)
            if self.act and i != self.layer_num - 1:
                X_a = self.act_layers[i](X_a)
            if self.bat and i != self.layer_num - 1:
                X_a = self.bat_layers[i](X_a)
            if self.dropout > 0 and i != self.layer_num - 1:
                X_a = F.dropout(X_a, self.dropout, training=self.training)

        if self.final_mlp:
            embeding_a = self.mlp(X_a)
        else:
            embeding_a = X_a

        ZP = self.P_layer(X_a)
        # X_a = F.relu(self.GCN_layers[0](A_a, X_a))
        # embeding_a = self.GCN_layers[1](A_a, X_a)
        # inputx = F.dropout(X_a, 0.1, training=self.training)
        # x = self.GCN_layers[0](A_a, inputx)
        # x = F.relu(x)
        # x = F.dropout(x, 0.1, training=self.training)
        # embeding_a = self.GCN_layers[1](A_a, x)
        # embeding_a = (embeding_a - embeding_a.mean(0)) / embeding_a.std(0)

        return embeding_a, ZP, S


class RGCN(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.cfg = args.cfg
        # if not os.path.exists(self.args.save_root):
        #     os.makedirs(self.args.save_root)
        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        # self.cfg.append(nb_classes)
        self.model = RGNN_Model(self.args.ft_size, cfg=self.cfg, final_mlp = nb_classes, gnn = self.args.gnn, dropout=self.args.random_aug_feature).to(self.args.device)

    def training(self):

        features = self.features.to(self.args.device)
        # graph_org = self.dgl_graph.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)
        number_neibor_list = splite_nerbor(self.adj_list[-1])
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
        ZP_list = []
        # features = F.normalize(features)
        for epoch in range(self.args.nb_epochs):
            self.model.train()
            optimiser.zero_grad()

            # A_a, X_a = process.RA(graph_org.cpu(), features, self.args.random_aug_feature,self.args.random_aug_edge)
            # A_a = A_a.add_self_loop().to(self.args.device)

            embeds, ZP, S = self.model(graph_org_torch, features)

            embeds_preds = torch.argmax(embeds, dim=1)
            
            train_embs = embeds[self.idx_train]
            val_embs = embeds[self.idx_val]
            test_embs = embeds[self.idx_test]

            GLR_loss = 0
            if epoch > 1:
                adj_label = get_similarity(ZP_list[-1])
                GLR_loss = self.args.beta * Ncontrast(adj_label, S)

            loss_reverse = self.args.gama * F.smooth_l1_loss(ZP, F.normalize(features))
            # features_nomal2 = ((a + 1) * features_nomal - 1) * (1 / a)
            # # loss_reverse = L2(output_fit, B)*0.0 + 0*F.smooth_l1_loss(output_fit, features_nomal) +a*(5*F.smooth_l1_loss(output_fit, features_nomal2) - 0*(output_fit.sum()/features_nomal.sum()))
            # # loss_reverse = torch.exp(-cos(F.normalize(output_fit), F.normalize(features_nomal2))).sum() / (N * N)
            # # loss_reverse = 100*F.smooth_l1_loss(F.normalize(output_fit), features_nomal2) + ((features_nomal - output_fit)*features_nomal).sum()/N
            # loss_reverse = b * F.smooth_l1_loss(output_fit, features_nomal2) + (
            #         (features_nomal - output_fit) * features_nomal).sum() / features_nomal.sum()
            loss = F.cross_entropy(train_embs, train_lbls) + GLR_loss +  loss_reverse
            ZP_list.append(ZP.detach())
            loss.backward()
            totalL.append(loss.item())
            optimiser.step()

            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0 :
                self.model.eval()
                # A_a, X_a = process.RA(graph_org.cpu(), features, 0, 0)
                # A_a = A_a.add_self_loop().to(self.args.device)
                embeds, ZP, S = self.model(graph_org_torch, features)
                val_acc = accuracy(embeds[self.idx_val], val_lbls)
                test_acc = accuracy(embeds[self.idx_test], test_lbls)
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