import time
from utils import process
from embedder import embedder_single
import os
from tqdm import tqdm
from evaluate import evaluate, accuracy
from models.Net import SUGRL_Fast, GCN_Fast
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


class RLGL(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.cfg = args.cfg

    def training(self):

        features = self.features.to(self.args.device)
        graph_org = self.dgl_graph.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)
        print("Started training...")
        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.cfg.append(nb_classes)
        model_critic = GCN_Fast(self.args.ft_size, cfg=self.cfg, final_mlp=0, gnn=self.args.gnn,
                         dropout=self.args.random_aug_feature).to(self.args.device)

        optimiser_critic = torch.optim.Adam(model_critic.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
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
            model_critic.train()
            optimiser_critic.zero_grad()
            embeds = model_critic(graph_org, features)
            embeds_preds = torch.argmax(embeds, dim=1)

            train_embs = embeds[self.idx_train]
            val_embs = embeds[self.idx_val]
            test_embs = embeds[self.idx_test]

            loss = F.cross_entropy(train_embs, train_lbls)

            loss.backward()
            totalL.append(loss.item())
            optimiser_critic.step()

            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0:
                model_critic.eval()
                embeds = model_critic(graph_org, features)
                val_acc = accuracy(embeds[self.idx_val], val_lbls)
                test_acc = accuracy(embeds[self.idx_test], test_lbls)
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