import torch
import time
import numpy as np
import copy
from models.embedder import embedder_single
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from models.Layers import act_layer
from evaluate import accuracy


class RecordeBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_end = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_end[:]


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
        self.buffer = RecordeBuffer()

    def forward(self, adj, feature_origin, labels=None):
        # TODO: every node with different number of neighbors, how to handle them with a matrix way.
        feature = copy.deepcopy(feature_origin)
        feature = self.get_embedding(feature)  # change the feature dimension to embedding dimension
        for i, col in enumerate(adj):
            idx = list(np.where(col.detach().cpu() != 0)[0])  # the neighbor index of node i
            idx.append(self.ending_idx)  # ending neighbor index
            # init the ending neighbor feature
            ending_neighbor_feature = torch.zeros(self.embedding_features).to(self.device)
            signal_neighbor_i = []
            ut_index = 0
            flag = False  # 选邻居了
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
                    if flag and labels is not None:
                        self.buffer.is_end[-1] = True
                    break

                # get the action by pi_theta
                at_distribution = F.softmax(self.get_action(hidden[ut_index]), dim=0)
                at_dist = Categorical(at_distribution)
                at = at_dist.sample()

                # calculate the reward
                hv = [feature_origin[u] for u in signal_neighbor_i]
                if labels is not None:
                    rt = self.get_reward(feature_origin[i], feature_origin[idx[ut_index]], hv, labels[i])
                    self.buffer.actions.append(at)
                    self.buffer.states.append(hidden[ut_index])
                    self.buffer.logprobs.append(at_dist.log_prob(at))
                    self.buffer.rewards.append(rt)
                    self.buffer.is_end.append(False)
                flag = True

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
                idx.pop(ut_index)  # del the selected node from the neighborhood set
        return feature

    def get_reward(self, xv, xu, neighbor, lb):
        r_xv = fc(self.get_rt(self.get_embedding((xv + xu) / 2)), lb)
        if len(neighbor) != 0:
            r_neighbor = fc(self.get_rt(self.get_embedding((xv + torch.cat(neighbor).reshape(-1, xv.shape[0])) / 2)),
                            lb)
        else:
            r_neighbor = fc(self.get_rt(self.get_embedding(xv)), lb)
        return r_xv / r_neighbor if r_neighbor.sum() != 0 else 0

    def evaluate(self, state, action):
        at_distribution = F.softmax(self.get_action(state), dim=0)
        dist = Categorical(at_distribution)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy


class GDP_Module(nn.Module):
    def __init__(self, input_features, layers, embedding_features, n_classes=7, lr=0.0003, gamma=0.95, K_epoch=1,
                 eps_clip=0.2, act='relu', device='cpu'):
        super(GDP_Module, self).__init__()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epoch
        self.device = device

        self.policy = Policy(input_features, layers, embedding_features, n_classes=n_classes, bias=False, act=act,
                             device=device)
        self.optimizer = torch.optim.Adam(params=self.policy.parameters(), lr=lr)
        self.policy_old = Policy(input_features, layers, embedding_features, n_classes=n_classes, bias=False, act=act,
                                 device=device)
        self.predict = nn.Linear(embedding_features, n_classes, bias=False)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()

    def update(self, adj, x, labels):
        self.policy_old(adj, x, labels)
        rewards = []
        discounted_reward = 0
        for reward, is_end in zip(reversed(self.policy_old.buffer.rewards), reversed(self.policy_old.buffer.is_end)):
            if is_end:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.policy_old.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.policy_old.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.policy_old.buffer.logprobs, dim=0)).detach().to(self.device)

        for i in range(self.K_epochs):
            logprobs, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = rewards
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards_norm - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy
            if i % 10 == 0:
                print(f"PPO Loss {loss.mean().item()}")
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.buffer.clear()

    def forward(self, adj, x):
        embedding = self.policy(adj, x)
        return self.predict(embedding)


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
        optimiser = torch.optim.Adam(self.model.predict.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
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
            # self.model.update(graph_org, features, self.labels)
            embeds = self.model(graph_org, features)
            loss = F.cross_entropy(embeds[self.idx_train], train_lbls)
            print(loss)
            loss.backward()
            totalL.append(loss.item())

            if epoch % 5 == 0 and epoch != 0:
                self.model.eval()
                with torch.no_grad():
                    embeds = self.model(graph_org, features)
                    val_acc = accuracy(embeds[self.idx_val], val_lbls)
                    test_acc = accuracy(embeds[self.idx_test], test_lbls)
                print(f"Epoch: {epoch} cls_loss: {sum(totalL) / len(totalL)}  val_acc: {val_acc} test_acc: {test_acc}")
                totalL = []
                stop_epoch = epoch
                if val_acc > best:
                    best = val_acc
                    output_acc = test_acc.item()
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if cnt_wait == self.args.patience:
                    break

        training_time = time.time() - start
        print("\t[Classification] ACC: {:.4f} | stop_epoch: {:}| training_time: {:.4f} ".format(
            output_acc, stop_epoch, training_time))
        return output_acc, training_time, stop_epoch
