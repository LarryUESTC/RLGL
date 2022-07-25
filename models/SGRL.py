import time
from utils import process
from embedder import embedder_single
import os
from tqdm import tqdm
from evaluate import evaluate, accuracy
from models.SUGRL_Fast import SUGRL_Fast, GCN_Fast
import numpy as np
import random as random
import torch.nn.functional as F
import torch
import torch.nn as nn
from dgl.nn import GraphConv, EdgeConv, GATConv

np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


class SGRL(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.cfg = args.cfg
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):

        features = self.features.to(self.args.device)
        graph_org = self.dgl_graph
        # adj_list = [adj.to(self.args.device) for adj in self.adj_list]
        I_target = torch.tensor(np.eye(self.cfg[-1])).to(self.args.device)

        print("Started training...")

        model = SUGRL_Fast(self.args.ft_size, cfg=self.cfg, final_mlp = self.cfg[-1]).to(self.args.device)

        optimiser = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay = self.args.wd)

        cnt_wait = 0
        best = 1e9
        stop_epoch = 0

        start = time.time()
        totalL = []

        for epoch in range(self.args.nb_epochs):
            model.train()
            optimiser.zero_grad()

            A_a, X_a = process.RA(graph_org.cpu(), features, self.args.random_aug_feature,self.args.random_aug_edge)
            A_b, X_b = process.RA(graph_org.cpu(), features, self.args.random_aug_feature,self.args.random_aug_edge)

            A_a = A_a.add_self_loop().to(self.args.device)
            A_b = A_b.add_self_loop().to(self.args.device)

            embeding_a, embeding_b = model(A_a, X_a, A_b, X_b)
            c1 = torch.mm(embeding_a.T, embeding_a) / self.args.nb_nodes
            c2 = torch.mm(embeding_b.T, embeding_b) / self.args.nb_nodes

            loss_c1 = (I_target - c1).pow(2).mean() + torch.diag(c1).mean()
            loss_c2 = (I_target - c2).pow(2).mean() + torch.diag(c2).mean()

            loss = 1 - self.args.alpha * F.cosine_similarity(embeding_a, embeding_b.detach(), dim=-1).mean()
            + self.args.beta * (loss_c1 + loss_c2)


            loss.backward()
            totalL.append(loss.item())
            optimiser.step()

            # early stop
            stop_epoch = epoch
            if loss < best:
                best = loss
                cnt_wait = 0
                # torch.save(model.state_dict(), 'saved_model/best_{}.pkl'.format(self.args.dataset))
            else:
                cnt_wait += 1
            if cnt_wait == self.args.patience:
                # print("Early stopped!")
                break
        training_time = time.time() - start
        # print("training time:{}s".format(training_time))
        # print("Evaluating...")
        model.eval()
        A_a, X_a = process.RA(graph_org.cpu(), features, 0, 0)
        A_a = A_a.add_self_loop().to(self.args.device)
        embeding = model.get_embedding(A_a, X_a)
        test_acc, test_acc_std, test_macro_f1s, test_macro_f1s_std, test_micro_f1s, test_micro_f1s_std, test_k1, test_k2, test_st = evaluate(embeding, self.idx_train, self.idx_val, self.idx_test, self.labels,
                                                    seed=self.args.seed, epoch=self.args.test_epo,
                                                    lr=self.args.test_lr)  # ,seed=seed
        # for l in totalL:
        #     print(l)
        return test_acc, test_acc_std, test_macro_f1s, test_macro_f1s_std, test_micro_f1s, test_micro_f1s_std, test_k1, test_k2, test_st,\
               training_time, stop_epoch


class SemiGRL(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.cfg = args.cfg
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):

        features = self.features.to(self.args.device)
        graph_org = self.dgl_graph.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)
        print("Started training...")
        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.cfg.append(nb_classes)
        model = GCN_Fast(self.args.ft_size, cfg=self.cfg, final_mlp = 0, gnn = self.args.gnn, dropout=self.args.random_aug_feature).to(self.args.device)

        optimiser = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay = self.args.wd)
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
            model.train()
            optimiser.zero_grad()

            # A_a, X_a = process.RA(graph_org.cpu(), features, self.args.random_aug_feature,self.args.random_aug_edge)
            # A_a = A_a.add_self_loop().to(self.args.device)

            embeds = model(graph_org, features)
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
                model.eval()
                # A_a, X_a = process.RA(graph_org.cpu(), features, 0, 0)
                # A_a = A_a.add_self_loop().to(self.args.device)
                embeds = model(graph_org, features)
                val_acc = accuracy(embeds[self.idx_val], val_lbls)
                test_acc = accuracy(embeds[self.idx_test], test_lbls)
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