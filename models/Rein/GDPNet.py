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
import setproctitle
import matplotlib.pyplot as plt
from tqdm import tqdm

setproctitle.setproctitle('zerorains')
# torch.autograd.set_detect_anomaly(True)



class RecordeBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_end = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_end[:]
        del self.values[:]


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
        # get the lk in every node k. lk is a score, so the output is 1, state-value function
        self.get_lk = nn.Linear(layers[-1], 1, bias=bias)
        # get the action value. action ={0,1}, so the output is 2
        self.get_action = nn.Linear(layers[-1], 2, bias=bias)
        # use to calculate the embedding of node v
        self.get_embedding = nn.Sequential(
            nn.Linear(input_features, embedding_features, bias=bias),
            nn.ReLU(inplace=True)
        )
        # using in reward calculate
        self.get_rt = nn.Sequential(
            nn.Linear(embedding_features, 64, bias=bias),
            act_layer(act),
            nn.Linear(64, n_classes, bias=bias)
        )
        self.buffer = RecordeBuffer()

    def forward(self, adj, features, labels=None, pretrain=False):
        # TODO: every node with different number of neighbors, how to handle them with a matrix way.
        feature_origin = copy.deepcopy(features)
        feature = self.get_embedding(features)  # change the feature dimension to embedding dimension
        if not pretrain:
            embedding = torch.zeros_like(feature)
            for i, col in tqdm(enumerate(adj)):
                idx = list(np.where(col.detach().cpu() != 0)[0])  # the neighbor index of node i
                idx.append(self.ending_idx)  # ending neighbor index
                # init the ending neighbor feature
                ending_neighbor_feature = torch.zeros(self.embedding_features).to(self.device)
                signal_neighbor_i = []
                ut_index = 0
                flag = False
                while ut_index != self.ending_idx:
                    # determine the neighborhood order
                    # concat the h_v and the h_ut
                    s = [torch.cat((feature[i], feature[v]))
                         if v > 0 else torch.cat((feature[i], ending_neighbor_feature))
                         for v in idx]
                    s = torch.stack(s, dim=0)
                    hidden = self.fc(s)
                    lk = self.get_lk(hidden)  # get the regret score l, it just like state-value function Q
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
                        self.buffer.actions.append(at.detach())
                        self.buffer.states.append(s[ut_index].detach())
                        self.buffer.logprobs.append(at_dist.log_prob(at).detach())
                        self.buffer.rewards.append(rt.detach())
                        self.buffer.is_end.append(False)
                        self.buffer.values.append(lk[ut_index].detach())
                    flag = True

                    if at == 1:
                        signal_neighbor_i.append(idx[ut_index])
                        hv.append(feature_origin[idx[ut_index]])
                    # update the representation
                    hv.append(feature_origin[i])
                    hv = torch.stack(hv, dim=0)
                    hv = self.get_embedding(hv.mean(dim=0))
                    hut = self.get_embedding(feature_origin[idx[ut_index]])
                    feature[i] = hv
                    feature[idx[ut_index]] = hut
                    embedding[i] = hv
                    embedding[idx[ut_index]] = hut
                    idx.pop(ut_index)  # del the selected node from the neighborhood set
            res = self.get_rt(embedding)
        else:
            res = self.get_rt(feature)
        return res

    def get_reward(self, xv, xu, neighbor, lb):
        r_xv = fc(self.get_rt(self.get_embedding((xv + xu) / 2)), lb)
        if len(neighbor) != 0:
            r_neighbor = fc(self.get_rt(self.get_embedding((xv + torch.stack(neighbor, dim=0)) / 2)),
                            lb)
        else:
            r_neighbor = fc(self.get_rt(self.get_embedding(xv)), lb)
        return r_xv / r_neighbor if r_neighbor.item() != 0 else r_xv

    def evaluate(self, state, action):
        hidden = self.fc(state)
        at_distribution = F.softmax(self.get_action(hidden), dim=0)
        dist = Categorical(at_distribution)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.get_lk(hidden)
        return state_value, action_logprobs, dist_entropy


class GDP_Module(nn.Module):
    def __init__(self, input_features, layers, embedding_features, n_classes=7, lr=0.005, gamma=0.99, gae_lambda=0.95,
                 K_epoch=40, eps_clip=0.2, act='relu', device='cpu'):
        super(GDP_Module, self).__init__()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.gae_lambda = gae_lambda
        self.device = device

        self.policy = Policy(input_features, layers, embedding_features, n_classes=n_classes, bias=False, act=act,
                             device=device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def update(self, adj, x, labels, pretrain=False, train_index=None):
        if pretrain:
            pred = self.policy(adj, x, pretrain=pretrain, labels=labels)
            if train_index is not None:
                loss = self.ce_loss(pred[train_index], labels[train_index])
            else:
                loss = self.ce_loss(pred, labels)
            print(f"pretrain loss: {loss.item()}")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.policy_old.load_state_dict(self.policy.state_dict())
            return
        res = self.policy(adj, x, labels)  # get reward state and caction (in polic.buffer)
        values = torch.cat(self.policy.buffer.values)
        reward_len = len(self.policy.buffer.rewards)
        advantage = torch.zeros(reward_len, dtype=torch.float32).to(self.device)

        # # complex calculate way
        # for t in tqdm(range(reward_len - 1)):
        #     discount = 1
        #     a_t = 0
        #     for k in range(t, reward_len - 1):
        #         a_t += discount * (self.policy.buffer.rewards[k] + self.gamma * values[k + 1] * (
        #                 1 - int(self.policy.buffer.is_end[k])) - values[k])
        #         discount *= self.gamma * self.gae_lambda
        #     advantage[t] = a_t

        # easy way to calculate the advantage by the follow equation
        # a_{t} = rewards_{t} + discount * values_{t+1} * (1-done_t) - values_{t} +  discount * gae_lambda * a_{t+1}
        for i in range(reward_len - 2, -1, -1):
            advantage[i] = self.policy.buffer.rewards[i] + self.gamma * values[i + 1] * (
                    1 - int(self.policy.buffer.is_end[i])) - values[i] + self.gamma * self.gae_lambda * advantage[i + 1]

        states = torch.stack(self.policy.buffer.states, dim=0).to(self.device)
        old_probs = torch.stack(self.policy.buffer.logprobs, dim=0).to(self.device)
        actions = torch.stack(self.policy.buffer.actions, dim=0).to(self.device)

        hidden = self.policy.fc(states)
        dist = Categorical(F.softmax(self.policy.get_action(hidden), dim=-1))
        critic_value = self.policy.get_lk(hidden)

        critic_value = torch.squeeze(critic_value)

        new_probs = dist.log_prob(actions)
        prob_ratio = (new_probs - old_probs).exp()

        weighted_probs = advantage * prob_ratio
        weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

        actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

        returns = advantage + values
        critic_loss = self.mse_loss(returns, critic_value)

        cls_loss = self.ce_loss(res[train_index], labels[train_index])

        loss = actor_loss + 0.5 * critic_loss + cls_loss
        rewards = sum(self.policy.buffer.rewards).item()
        print(f"PPO Loss: {loss} rewards: {rewards}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.policy.buffer.clear()

        return rewards

    def forward(self, adj, x):
        res = self.policy(adj, x)
        return res


class GDPNet(embedder_single):
    def __init__(self, args):
        args.device = 'cuda:1'
        super(GDPNet, self).__init__(args)
        self.args = args
        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.model = GDP_Module(self.features.shape[-1], self.args.cfg, self.args.feature_dimension, act='relu',
                                n_classes=nb_classes, device=self.args.device).to(self.args.device)

    def training(self):
        features = self.features.to(self.args.device)
        graph_org = self.adj_list[-1].to(self.args.device)
        self.labels = self.labels.to(self.args.device)
        print("Started training...")
        train_lbls = self.labels[self.idx_train]
        val_lbls = self.labels[self.idx_val]
        test_lbls = self.labels[self.idx_test]

        cnt_wait = 0
        best = 1e-9
        output_acc = 1e-9
        stop_epoch = 0

        start = time.time()
        rewards = []

        for epoch in range(500, self.args.pretrain_epochs + self.args.nb_epochs):
            self.model.train()
            if epoch < self.args.pretrain_epochs:
                self.model.update(graph_org, features, self.labels, pretrain=True, train_index=self.idx_train)
            else:
                print("*" * 15 + f"  Epoch {epoch - self.args.pretrain_epochs}  " + "*" * 15)
                reward = self.model.update(graph_org, features, self.labels, pretrain=False, train_index=self.idx_train)
                rewards.append(reward)

                plt.plot(range(len(rewards)), rewards)
                plt.savefig("rewards.png")
            if epoch >= self.args.pretrain_epochs and epoch % 5 == 0 and epoch != 0:
                self.model.eval()
                with torch.no_grad():
                    embeds = self.model(graph_org, features)
                    val_acc = accuracy(embeds[self.idx_val], val_lbls)
                    test_acc = accuracy(embeds[self.idx_test], test_lbls)
                print(f"Epoch: {epoch - self.args.pretrain_epochs}  val_acc: {val_acc} test_acc: {test_acc}")
                stop_epoch = epoch
                if val_acc > best:
                    best = val_acc
                    output_acc = test_acc.item()
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                # if cnt_wait == self.args.patience:
                #     break

        training_time = time.time() - start
        print("\t[Classification] ACC: {:.4f} | stop_epoch: {:}| training_time: {:.4f} ".format(
            output_acc, stop_epoch, training_time))

        return output_acc, training_time, stop_epoch
