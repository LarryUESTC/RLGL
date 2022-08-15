import torch
import time
import numpy as np
import copy
from models.embedder import embedder_single
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from models.Layers import act_layer
from tqdm import tqdm


def fc(x, lb):
    # TODO: the fc function in Eq.5
    return x.argmax(dim=-1).eq(lb).sum()


class Policy(nn.Module):
    def __init__(self, input_features, layers, embedding_features, n_classes=7, bias=True, act='relu', device='cpu'):
        super(Policy, self).__init__()
        self.device = device
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
        # get the lk in every node k. lk is a score, so the output is 1
        self.get_lk = nn.Linear(layers[-1], 1, bias=bias)
        # get the action value. action ={0,1}, so the output is 2
        self.get_action = nn.Linear(layers[-1], 2, bias=bias)
        # use to calculate the embedding of node v
        self.get_embedding = nn.Sequential(
            nn.Linear(input_features, embedding_features, bias=bias),
            act_layer(act)
        )
        self.get_rt = nn.Sequential(
            nn.Linear(embedding_features, 64, bias=bias),
            act_layer(act),
            nn.Linear(64, n_classes, bias=bias)
        )

    def forward(self, adj, feature_origin, labels):
        # TODO: every node with different number of neighbors, how to handle them with a matrix way.
        feature = copy.deepcopy(feature_origin)
        feature = self.get_embedding(feature)  # change the feature dimension to embedding dimension
        embedding = torch.zeros_like(feature)
        signal_neighbor = []
        for i, col in enumerate(adj):
            idx = list(np.where(col.detach().cpu() != 0)[0])  # the neighbor index of node i
            idx.append(self.ending_idx)  # ending neighbor index
            # init the ending neighbor feature
            ending_neighbor_feature = torch.zeros(self.embedding_features).to(self.device)
            signal_neighbor_i = []
            signal_neighbor_recorde = []
            ut_index = 0
            while ut_index != self.ending_idx:
                # determine the neighborhood order
                # concat the h_v and the h_ut
                s = [torch.cat((feature[i], feature[v]))
                     if v > 0 else torch.cat((feature[i], ending_neighbor_feature))
                     for v in idx]
                s = torch.cat(s).reshape(-1, self.embedding_features * 2)
                hidden = self.fc(s)
                lk = self.get_lk(hidden)  # get the regret score l
                ut_distribution = F.softmax(lk, dim=0)
                # get the ut~softmax([l1,...,le,...,lk])
                ut_dist = Categorical(ut_distribution.T)
                ut_index = ut_dist.sample().item()  # sampling
                if idx[ut_index] == self.ending_idx:  # if sample the ending node, stop it
                    break

                # get the action by pi_theta
                at_distribution = F.softmax(self.get_action(hidden[ut_index]), dim=0)
                at_dist = Categorical(at_distribution)
                at = at_dist.sample()

                # calculate the reward
                hv = [feature_origin[u] for u in signal_neighbor_i]
                rt = self.get_reward(feature_origin[i], feature_origin[idx[ut_index]], hv, labels[i])
                signal_neighbor_recorde.append([idx[ut_index], at, rt])

                if at == 1:
                    signal_neighbor_i.append(idx[ut_index])
                    hv.append(feature_origin[idx[ut_index]])
                # update the representation
                hv.append(feature_origin[i])
                hv = torch.cat(hv).reshape(-1, feature_origin.shape[-1])
                hv = self.get_embedding(hv.mean(dim=0))
                hut = self.get_embedding(feature_origin[idx[ut_index]])
                feature[i] = hv
                feature[idx[ut_index]] = hut
                # TODO:区分每个节点的embedding更新时使用的特征
                embedding[i] = hv
                embedding[idx[ut_index]] = hut

                del idx[ut_index]  # del the selected node from the neighborhood set
            signal_neighbor.append(signal_neighbor_recorde)
        return embedding, signal_neighbor

    def get_reward(self, xv, xu, neighbor, lb):
        r_xv = fc(self.get_rt(self.get_embedding((xv + xu) / 2)), lb)
        if len(neighbor) != 0:
            r_neighbor = fc(self.get_rt(self.get_embedding((xv + torch.cat(neighbor).reshape(-1, xv.shape[0])) / 2)),
                            lb)
        else:
            r_neighbor = fc(self.get_rt(self.get_embedding(xv)), lb)
        return r_xv / r_neighbor if r_neighbor.sum() != 0 else 0


class GDP_Module(nn.Module):
    def __init__(self, input_features, layers, embedding_features, n_classes=7, act='relu', device='cpu'):
        super(GDP_Module, self).__init__()
        self.policy = Policy(input_features, layers, embedding_features, n_classes=n_classes, bias=False, act=act,
                             device=device)

    def forward(self, adj, x, labels):
        embedding, action_recorde = self.policy(adj, x, labels)
        return x


class GDPNet(embedder_single):
    def __init__(self, args):
        super(GDPNet, self).__init__(args)
        self.args = args
        # args.device = 'cpu'
        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.model = GDP_Module(self.features.shape[-1], self.args.cfg, self.args.feature_dimension, act='relu',
                                n_classes=nb_classes, device=self.args.device).to(self.args.device)

    def training(self):
        features = self.features.to(self.args.device)
        graph_org = self.adj_list[-1].to(self.args.device)
        self.labels = self.labels.to(self.args.device)
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

            embeds = self.model(graph_org, features, self.labels)
            break
