import torch
import time
import numpy as np
import copy
from models.embedder import embedder_single
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from models.Layers import act_layer, GNN_Model
# from models.NodeClas.Semi_SELFCONS import GAT_selfCon_Trans
from models.Rein.RLG import KDloss_0
from evaluate import accuracy
import setproctitle
import random
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


def get_A_r(adj, r):
    adj_label = adj
    for i in range(r - 1):
        adj_label = adj_label @ adj

    return adj_label


def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0]).cuda(3)
    x_sum = torch.sum(x ** 2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


def Ncontrast(x_dis, adj_label, tau=1.):
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
    return loss


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
        self.get_lk = nn.Sequential(
            nn.Linear(layers[-1], 1, bias=bias)
        )
        # get the action value. action ={0,1}, so the output is 2
        self.get_action = nn.Sequential(
            nn.Linear(layers[-1], 2, bias=bias)
        )
        # use to calculate the embedding of node v
        self.get_embedding = nn.Sequential(
            nn.Linear(input_features, embedding_features, bias=bias),
            nn.ReLU(inplace=True)
        )
        # using in reward calculate
        # self.get_rt = nn.Sequential(
        #     nn.Linear(embedding_features, 64, bias=bias),
        #     act_layer(act),
        #     nn.Linear(64, n_classes, bias=bias)
        # )
        self.buffer = RecordeBuffer()

    def pi(self, x):
        z = self.get_embedding(x)
        z = self.get_action(z)
        prob = F.softmax(z)
        return prob

    def v(self, x):
        z = self.get_embedding(x)
        v = self.get_lk(z)
        return v

    # def forward(self, adj, features, labels=None, pretrain=False):
    #     feature_origin = copy.deepcopy(features)
    #     neighbor_num = 0.0
    #     done = False
    #     T_horizon = 5
    #     s = self.env_reset()
    #     while not done:
    #         for t in range(T_horizon):
    #             prob = self.pi(torch.from_numpy(s).float())
    #             m = Categorical(prob)
    #             a = m.sample().item()
    #             s_prime, r, done, info = env.step(a)
    #
    #             model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), done))
    #             s = s_prime
    #
    #             score += r
    #             if done:
    #                 break
    #
    #     return

    def get_reward(self, xv, xu, neighbor, lb):
        r_xv = fc(self.get_rt(self.get_embedding((xv + xu) / 2)), lb)
        if len(neighbor) != 0:
            r_neighbor = fc(self.get_rt(self.get_embedding((xv + torch.stack(neighbor, dim=0)) / 2)),
                            lb)
        else:
            r_neighbor = fc(self.get_rt(self.get_embedding(xv)), lb)
        return r_xv / r_neighbor if r_neighbor.item() != 0 else r_xv

    def get_reward1(self, x, label):
        embeds = self.get_rt(self.get_embedding(x))
        reward = -KDloss_0(embeds, label.unsqueeze(dim=0))
        return reward

    def get_reward2(self, x, adj):
        if len(x.shape) < 2 or (len(x.shape) >= 2 and x.shape[0] == 1):
            return torch.tensor(-100)
        x_dis = get_feature_dis(self.get_embedding(x))
        label_adj = get_A_r(adj, 4)
        reward = -Ncontrast(x_dis, label_adj)
        return reward

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
        batch_size = 256
        batch_idx = random.sample(list(range(0, x.shape()[0]-1)), batch_size)


        res, nodes = self.policy(adj, x, labels)  # get reward state and caction (in polic.buffer)
        values = torch.cat(self.policy.buffer.values)
        reward_len = len(self.policy.buffer.rewards)
        advantage = torch.zeros(reward_len, dtype=torch.float32).to(self.device)


        for i in range(reward_len - 2, -1, -1):
            advantage[i] = self.policy.buffer.rewards[i] + self.gamma * values[i + 1] * (
                    1 - int(self.policy.buffer.is_end[i])) - values[i] + self.gamma * self.gae_lambda * advantage[
                               i + 1] * (1 - int(self.policy.buffer.is_end[i]))

        states = torch.stack(self.policy.buffer.states, dim=0).to(self.device)
        old_probs = torch.stack(self.policy.buffer.logprobs, dim=0).to(self.device)
        actions = torch.stack(self.policy.buffer.actions, dim=0).to(self.device)

        for i in range(self.K_epoch):
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

            loss = actor_loss + 0.5 * critic_loss

            print(f"RL Epoch: {i} PPO Loss: {loss}  node_num: {nodes}")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        rewards = sum(self.policy.buffer.rewards).item()
        self.policy.buffer.clear()

        return rewards

    def forward(self, adj, x):
        res, _ = self.policy(adj, x)
        return res


class GDPNet2(embedder_single):
    def __init__(self, args):

        # args.device = 'cuda:3'
        super(GDPNet2, self).__init__(args)
        self.args = args
        self.fake_labels = None
        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.model = GDP_Module(self.features.shape[-1], self.args.cfg, self.args.feature_dimension, act='relu',
                                n_classes=nb_classes, device=self.args.device).to(self.args.device)
        self.env_model = GNN_Model(self.args.ft_size, cfg=[128,16].append(nb_classes), final_mlp = 0, gnn = self.args.gnn, dropout=self.args.random_aug_feature).to(self.args.device)

    def training(self):
        features = self.features.to(self.args.device)
        graph_org = self.adj_list[-1].to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)
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
        acces = []

        # pre_model = GAT_selfCon_Trans(features.shape[-1], cfg=[256, 7], final_mlp=0, dropout=0.2, nheads=8,
        #                               Trans_layer_num=2).to(self.args.device)
        # pre_optimizer = torch.optim.Adam(pre_model.parameters(), lr=0.0005, weight_decay=0.0005)

        env_embedding = self.pre_training()
        adj_label = get_A_r(graph_org_torch, 4)
        for epoch in range(self.args.nb_epochs):
            self.model.train()

            print("*" * 15 + f"  Epoch {epoch - self.args.pretrain_epochs}  " + "*" * 15)
            reward = self.model.update(graph_org_torch, features, self.labels, train_index=self.idx_train)
            rewards.append(reward)

            if epoch % 5 == 0 and epoch != 0:
                self.model.eval()
                with torch.no_grad():
                    embeds = self.model(graph_org, features)
                    val_acc = accuracy(embeds[self.idx_val], val_lbls)
                    test_acc = accuracy(embeds[self.idx_test], test_lbls)
                acces.append(test_acc.item())
                plt.plot(range(len(acces)), acces)
                plt.savefig("acc.png")
                plt.cla()
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

    def pre_training(self):

        features = self.features.to(self.args.device)
        # graph_org = self.dgl_graph.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)
        print("Started training...")

        optimiser = torch.optim.Adam(self.env_model.parameters(), lr=0.01, weight_decay=5e-4)
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
        Last_embedding = None
        for epoch in range(self.args.nb_epochs):
            self.env_model.train()
            optimiser.zero_grad()

            embeds = self.env_model(graph_org_torch, features)
            embeds_preds = torch.argmax(embeds, dim=1)

            train_embs = embeds[self.idx_train]
            val_embs = embeds[self.idx_val]
            test_embs = embeds[self.idx_test]

            loss = F.cross_entropy(train_embs, train_lbls)

            loss.backward()
            totalL.append(loss.item())
            optimiser.step()

            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0:
                self.env_model.eval()
                # A_a, X_a = process.RA(graph_org.cpu(), features, 0, 0)
                # A_a = A_a.add_self_loop().to(self.args.device)
                embeds = self.env_model(graph_org_torch, features)
                Last_embedding = embeds.detach()
                val_acc = accuracy(embeds[self.idx_val], val_lbls)
                test_acc = accuracy(embeds[self.idx_test], test_lbls)

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
        return Last_embedding

