import torch
import time
import numpy as np
import copy
from models.embedder import embedder_single
from torch import nn
from torch.nn import functional as F
from models.Layers import act_layer


class Policy(nn.Module):
    def __init__(self, input_features, layers, embedding_features, bias=True, act='relu'):
        super(Policy, self).__init__()
        self.embedding_features = embedding_features
        self.ending_idx = -233
        self.fc = nn.ModuleList([])
        # input: cat(h_v,h_ut)  out: hidden dimension1
        self.fc += nn.Sequential(
            nn.Linear(embedding_features * 2, layers[0], bias=bias),
            act_layer(act)
        )
        for i in range(len(layers) - 1):
            self.fc += [
                nn.Sequential(
                    nn.Linear(layers[i], layers[i + 1], bias=bias),
                    act_layer(act),
                )
            ]
        self.fc = nn.Sequential(*self.fc)
        # get the lk in every node k. lk is a score, so the output is 1 dimension
        self.get_lk = nn.Linear(layers[-1], 1, bias=bias)
        # get the action value. action ={0,1}, so the output is 2 dimension
        self.get_action = nn.Linear(layers[-1], 2, bias=bias)
        # use to calculate the embedding of node v
        self.get_embedding = nn.Sequential(
            nn.Linear(input_features, embedding_features, bias=bias),
            act_layer(act)
        )

    def forward(self, adj, feature_origin):
        # TODO: every node with different number of neighbors, how to handle them with a matrix way.
        feature = copy.deepcopy(feature_origin)
        feature = self.get_embedding(feature)  # change the feature dimension to embedding dimension
        signal_neighbor = []
        for i, col in enumerate(adj):
            idx = list(np.where(col != 0)[0])  # the neighbor index of node i
            idx.append(self.ending_idx)  # ending neighbor index
            ending_neighbor_feature = torch.zeros(self.embedding_features)  # init the ending neighbor feature
            # concat the h_v and the h_ut
            s = [torch.cat((feature[i], feature[v]))
                 if v > 0 else torch.cat((feature[i], ending_neighbor_feature))
                 for v in idx]
            s = torch.cat(s).reshape(-1, self.embedding_features * 2)
            lk = self.get_lk(self.fc(s))  # get the regret score l
            signal_neighbor_i = []
            ut_index = 0
            while ut_index != self.ending_idx:
                ut_distribution = F.softmax(lk, dim=0)
                ut_distribution = ut_distribution.reshape(-1).detach().numpy()  # get the ut~softmax([l1,...,le,...,lk])
                ut_index = np.random.choice(range(len(idx)), p=ut_distribution)  # sampling
                if idx[ut_index] == self.ending_idx:  # if sample the ending node, stop it
                    break
                # get the action by pi_theta
                at_distribution = F.softmax(self.get_action(self.fc(s[ut_index])), dim=0)
                at_distribution = at_distribution.detach().numpy()
                at = np.random.choice(range(2), p=at_distribution)
                if at == 1:
                    signal_neighbor_i.append(idx[ut_index])

                # update the representation
                hv = [feature_origin[u] for u in signal_neighbor_i]
                hv.append(feature_origin[i])
                hv = torch.cat(hv).reshape(-1, feature_origin.shape[-1])
                hv = self.get_embedding(hv.mean(dim=0))
                feature[idx[ut_index]] = hv

                del idx[ut_index]  # del the selected node from the neighborhood set
                s = s[range(s.shape[0]) != ut_index]
                lk = lk[range(lk.shape[0]) != ut_index]
        return feature


class GDP_Module(nn.Module):
    def __init__(self, input_features, layers, embedding_features, act='relu'):
        super(GDP_Module, self).__init__()
        self.policy = Policy(input_features, layers, embedding_features, bias=False, act=act)

    def forward(self, adj, x):
        x = self.policy(adj, x)
        return x


class GDPNet(embedder_single):
    def __init__(self, args):
        super(GDPNet, self).__init__(args)
        self.args = args
        args.device = 'cpu'
        self.model = GDP_Module(self.features.shape[-1], self.args.cfg, self.args.feature_dimension, act='relu')

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
