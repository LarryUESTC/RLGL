from models.embedder import embedder_single
import torch
from torch import nn
from torch.nn import functional as F
from models.Layers import act_layer
import time
import numpy as np


class Policy(nn.Module):
    def __init__(self, input_features, layers, bias=True, act='relu'):
        super(Policy, self).__init__()
        self.fc = nn.ModuleList([])
        self.fc += nn.Sequential(nn.Linear(2 * input_features, layers[0], bias=bias), act_layer(act))
        for i in range(len(layers) - 1):
            self.fc += [
                nn.Sequential(
                    nn.Linear(layers[i], layers[i + 1], bias=bias),
                    act_layer(act),
                )
            ]
        self.fc = nn.Sequential(*self.fc)
        self.get_lk = nn.Sequential(
            nn.Linear(layers[-1], 1, bias=False),
            nn.Softmax(dim=0)
        )

    def forward(self, adj, feature):
        for i, col in enumerate(adj):
            idx = list(np.where(col != 0)[0])
            idx.append(-233)  # ending neighbor index
            ending_neighbor_feature = torch.zeros((feature.shape[-1]))
            s = [torch.cat((feature[i], feature[v]))
                 if v > 0 else torch.cat((feature[i], ending_neighbor_feature))
                 for v in idx]
            s = torch.cat(s).reshape(-1, feature.shape[-1] * 2)
            lk = self.get_lk(self.fc(s))
            hat_neighbor = np.random.choice(idx, lk.detach().numpy())
            break

        pass


class GDP_Module(nn.Module):
    def __init__(self, input_features, layers, act='relu'):
        super(GDP_Module, self).__init__()
        self.policy = Policy(input_features, layers, bias=False, act=act)

    def forward(self, adj, x):
        x = self.policy(adj, x)
        return x


class GDPNet(embedder_single):
    def __init__(self, args):
        super(GDPNet, self).__init__(args)
        self.args = args
        args.device = 'cpu'
        self.model = GDP_Module(self.features.shape[-1], self.args.cfg, act='relu')

    def training(self):
        features = self.features
        graph_org = self.adj_list[-1]
        print("Started training...")

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        train_lbls = self.labels[self.idx_train]
        val_lbls = self.labels[self.idx_val]
        test_lbls = self.labels[self.idx_test]

        cnt_wait = 0
        best = 1e-9
        output_acc = 1e-9
        stop_epoch = 0

        start = time.time()
        totalL = []

        for epoch in range(self.args.nb_epochs):
            self.model.train()
            optimiser.zero_grad()

            embeds = self.model(graph_org, features)
            break
