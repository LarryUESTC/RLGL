import torch
import time
import numpy as np
import copy
from models.embedder import embedder_single
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from models.Layers import act_layer, GNN_Model, GNN_pre_Model
# from models.NodeClas.Semi_SELFCONS import GAT_selfCon_Trans
from models.Rein.RLG import KDloss_0
from evaluate import accuracy
import setproctitle
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter

setproctitle.setproctitle('PLLL')


# torch.autograd.set_detect_anomaly(True)


class RecordeBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.states_next = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_end = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.states_next[:]
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
    mask = torch.eye(x_dis.shape[0]).to(x.device)
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
        self.input_features = input_features
        self.embedding_features = embedding_features
        self.ending_idx = -233
        self.writer_tb = SummaryWriter()
        self.fc = nn.ModuleList([])
        self.fc += nn.Sequential(
            nn.Linear(input_features * 3, layers[0], bias=bias),
            act_layer(act),
            nn.BatchNorm1d(layers[0])
        )
        for i in range(len(layers) - 1):
            self.fc += [
                nn.Sequential(
                    nn.Linear(layers[i], layers[i + 1], bias=bias),
                    act_layer(act),
                    nn.BatchNorm1d(layers[i + 1])
                )
            ]
        self.fc = nn.Sequential(*self.fc)

        # get the lk in every node k. lk is a score, so the output is 1, state-value function
        self.get_lk = nn.Sequential(
            nn.Linear(layers[-1], layers[-1], bias=bias),
            act_layer(act),
            nn.BatchNorm1d(layers[-1]),
            nn.Linear(layers[-1], 1, bias=bias)
        )
        # get the action value. action ={0,1}, so the output is 2
        self.get_action = nn.Sequential(
            nn.Linear(layers[-1], layers[-1], bias=bias),
            act_layer(act),
            nn.BatchNorm1d(layers[-1]),
            nn.Linear(layers[-1], 2, bias=bias)
        )
        # use to calculate the embedding of node v

        self.buffer = RecordeBuffer()

    def pi(self, s):
        z = self.fc(s)
        z = self.get_action(z)
        prob = F.softmax(z, dim=-1)
        return prob

    def v(self, s):
        #todo
        s= torch.cat([s[:, :128 * 2], s[:, 128 * 2:]*0], dim=-1)
        z = self.fc(s)
        v = self.get_lk(z)
        return v

    def get_reward_1(self, env_embedding_dis, consider_idx, at, adj_label):
        reward = (env_embedding_dis[[*range(0, len(consider_idx))], consider_idx] - env_embedding_dis.mean(dim = 1)*1.5) \
                 * (at*2-1)
        return reward

    def get_reward_2(self, env_embedding_dis, consider_idx, at, adj_label):
        reward = (adj_label[[*range(0, len(consider_idx))], consider_idx]  - adj_label.mean(dim = 1))* (at*2-1)
        # reward +=
        return reward*100

    def get_reward_3(self, env_embedding_dis, consider_idx, at, adj_label):
        reward = (adj_label[[*range(0, len(consider_idx))], consider_idx]  - adj_label.mean(dim = 1))* (at*7-1)
        # reward +=
        return reward*100

    def forward(self, adj, adj_wo_I, adj_wo_N, features, labels, env_embedding, adj_label, epoch):
        # feature_origin = copy.deepcopy(features)
        env_embedding = F.softmax(env_embedding)
        env_embedding_dis = get_feature_dis(env_embedding)
        adj_env = copy.deepcopy(adj_wo_N)
        # adj_env = torch.eye(adj_wo_N.size()[0]).to(adj_wo_N.device)
        embeddings = copy.deepcopy(features)  # change the feature dimension to embedding dimension

        adj_env_N = F.normalize(adj_env, p=1)
        embeddings_neibor = torch.mm(adj_env_N, embeddings)

        #todo select self or neibor?
        batch_size = features.size()[0]
        consider_idx = random.sample(list(range(0, batch_size)), batch_size)
        embeddings_consider = embeddings[consider_idx]

        s = torch.cat([embeddings, embeddings_neibor, embeddings_consider], dim=-1)
        T_horizon = 5
        ac_acc_list = []
        for t in range(T_horizon):
            z = self.fc(s)
            lk = self.get_action(z)
            at_distribution = F.softmax(lk, dim=-1)
            at_dist = Categorical(at_distribution)
            at = at_dist.sample()

            adj_env[[*range(0, batch_size)], consider_idx] = at+0.0
            adj_env_N = F.normalize(adj_env, p=1)
            rt = self.get_reward_1(env_embedding_dis, consider_idx, at, adj_label)

            at_acc = ((labels == labels[consider_idx]) * at).sum() / at.sum()
            ac_acc_list.append(at_acc)
            print(f"A-{t}: {at_acc}/{at.sum()}|", end="")
            self.writer_tb.add_scalar('global_acc_'+str(t), at_acc, epoch)
            self.writer_tb.add_scalar('global_num_' + str(t), at.sum(), epoch)

            embeddings_neibor_next = torch.mm(adj_env_N, embeddings)
            #todo select self or neibor?
            consider_idx_next = random.sample(list(range(0, batch_size)), batch_size)
            embeddings_consider_next = embeddings[consider_idx_next]
            s_next = torch.cat([embeddings, embeddings_neibor_next, embeddings_consider_next], dim=1)

            self.buffer.actions.append(at.detach())
            self.buffer.states.append(s.detach())
            self.buffer.states_next.append(s_next.detach())
            self.buffer.logprobs.append(at_dist.log_prob(at).detach())
            self.buffer.rewards.append(rt.detach())
            self.writer_tb.add_scalar('reward_' + str(t), rt.detach().mean().item(), epoch)
            self.buffer.is_end.append(False)

            consider_idx = consider_idx_next
            embeddings_consider = embeddings_consider_next
            s = s_next

        print("")
        return

    def evlue(self, adj, adj_wo_I, adj_wo_N, features, labels, env_embedding, adj_label):
        # feature_origin = copy.deepcopy(features)

        # env_embedding_dis = get_feature_dis(env_embedding)
        adj_env = copy.deepcopy(adj_wo_N)

        embeddings = copy.deepcopy(features)  # change the feature dimension to embedding dimension

        adj_env_N = F.normalize(adj_env, p=1)
        embeddings_neibor = torch.mm(adj_env_N, embeddings)

        # todo select self or neibor?
        batch_size = features.size()[0]
        for consider_idx in range(batch_size):
            num_idx = adj_env.sum(-1)[consider_idx]
            end_neibor = 5
            if num_idx > end_neibor:
                pass
            else:
                embeddings_i = embeddings[consider_idx].repeat(batch_size, 1)
                embeddings_neibor_i = embeddings_neibor[consider_idx].repeat(batch_size, 1)
                embeddings_consider_i = embeddings
                s = torch.cat([embeddings_i, embeddings_neibor_i, embeddings_consider_i], dim=-1)
                s_v = torch.cat([embeddings_i, (adj_env.sum(-1)[consider_idx] * embeddings_neibor_i + embeddings_consider_i) / (1 +adj_env.sum(-1)[consider_idx]), embeddings_consider_i], dim=-1)
                v = self.v(s_v)
                z = self.fc(s)
                lk = self.get_action(z)
                at_distribution = F.softmax(lk, dim=-1)

                at = at_distribution[:, 1].topk(k=int(5 - num_idx.item() + 1), dim=0, largest=True, sorted=True)[1]
                adj_env[consider_idx, at] = 1.0

                # max_idx = at_distribution[:, 1].topk(k=20, dim=0, largest=True, sorted=True)[1]
                # at = max_idx[v[max_idx].topk(k=int(5 - num_idx.item() + 1), dim=0, largest=True, sorted=True)[1]].squeeze()
                # adj_env[consider_idx, at] = 1.0
                # adj_env[at, consider_idx] = 1.0

        # adj_env_N = F.normalize(adj_env, p=1)

        print("")
        return adj_env


class GDP_Module(nn.Module):
    def __init__(self, input_features, layers, embedding_features, n_classes=7, lr=0.0005, gamma=0.99, gae_lambda=0.95,
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

    def make_batch(self):
        a = torch.stack(self.policy.buffer.actions)
        s = torch.stack(self.policy.buffer.states)
        s_prime = torch.stack(self.policy.buffer.states_next)
        prob_a = torch.stack(self.policy.buffer.logprobs)
        r = torch.stack(self.policy.buffer.rewards)
        # done_mask = torch.cat(self.policy.buffer.is_end)
        self.policy.buffer.clear()

        return a, s, s_prime, r, prob_a

    def update(self, adj, adj_wo_I, adj_wo_N, x, labels, env_embedding, adj_label, epoch, pretrain=False, train_index=None):
        self.policy(adj, adj_wo_I, adj_wo_N, x, labels, env_embedding, adj_label, epoch)  # get reward state and caction (in polic.buffer)
        a, s, s_prime, r, prob_a = self.make_batch()
        learning_rate = 0.0005
        gamma = 0.98
        lmbda = 0.95
        eps_clip = 0.1
        K_epoch = 3
        T_horizon = s.size()[0]
        batch_size = s.size()[1]

        for i in range(K_epoch):
            td_target = r.view(T_horizon*batch_size, -1) + gamma * self.policy.v(s_prime.view(T_horizon*batch_size, -1))
            delta = td_target - self.policy.v(s.view(T_horizon*batch_size, -1))
            delta = delta.detach().view(T_horizon, batch_size, -1)

            advantage_lst = []
            advantage = 0.0
            for delta_t in torch.flip(delta, dims=[0]):
                advantage = gamma * lmbda * advantage + delta_t
                advantage_lst.append(advantage)
            advantage_lst.reverse()
            advantage = torch.stack(advantage_lst).view(T_horizon*batch_size, -1)
            # advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.policy.pi(s.view(T_horizon*batch_size, -1))
            pi_a = pi.view(T_horizon,batch_size, -1).gather(-1,a.unsqueeze(dim = -1)).squeeze().view(T_horizon*batch_size, -1)
            ratio = torch.exp(torch.log(pi_a) - prob_a.view(T_horizon*batch_size, -1))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.policy.v(s.view(T_horizon*batch_size, -1)), td_target.detach())

            # print(f"RL Epoch: {i} PPO Loss: {loss.mean()} ")
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        return

    def forward(self, adj, adj_wo_I, adj_wo_N, x, labels, env_embedding, adj_label, pretrain=False, train_index=None):
        adj_new = self.policy.evlue(adj, adj_wo_I, adj_wo_N, x, labels, env_embedding, adj_label)
        return adj_new


class RLGDQN(embedder_single):
    def __init__(self, args):

        # args.device = 'cuda:3'
        super(RLGDQN, self).__init__(args)
        self.args = args
        self.fake_labels = None
        self.nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.model = GDP_Module(128, self.args.cfg, self.args.feature_dimension, act='gelu',
                                n_classes=self.nb_classes, device=self.args.device).to(self.args.device)

        self.env_model = GNN_pre_Model(self.args.ft_size, cfg=[128, 16, self.nb_classes], final_mlp = 0, gnn = self.args.gnn, dropout=self.args.random_aug_feature).to(self.args.device)

    def training(self):
        features = self.features.to(self.args.device)
        graph_org = self.adj_list[-1].to(self.args.device)
        graph_org_N = self.adj_list[1].to(self.args.device)
        graph_org_NI = self.adj_list[0].to(self.args.device)
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

        env_embedding, env_embedding_first = self.pre_training()

        adj_label = get_A_r(graph_org_NI, 4)

        for epoch in range(self.args.nb_epochs):
            self.model.train()
            print("*" * 15 + f"  Epoch {epoch}  " + "*" * 15)
            reward = self.model.update(graph_org_NI, graph_org_N, graph_org, env_embedding_first, self.labels, env_embedding, adj_label, epoch, train_index=self.idx_train)
            rewards.append(reward)

            if epoch % 20 == 0 and epoch != 0:
                self.model.eval()
                adj_new = self.model(graph_org_NI, graph_org_N, graph_org, env_embedding_first, self.labels, env_embedding, adj_label, train_index=self.idx_train)
                # graph_org_torch = F.normalize(adj_new + torch.eye(adj_new.size()[0]).to(adj_new.device), p= 1)
                graph_org_torch = F.normalize(adj_new, p=1) + torch.eye(adj_new.size()[0]).to(adj_new.device)
                env_model = GNN_Model(self.args.ft_size, cfg=[16, self.nb_classes], final_mlp = 0, gnn = self.args.gnn, dropout=self.args.random_aug_feature).to(self.args.device)
                features = self.features.to(self.args.device)
                optimiser = torch.optim.Adam(env_model.parameters(), lr=0.01, weight_decay=5e-4)
                xent = nn.CrossEntropyLoss()
                train_lbls = self.labels[self.idx_train]
                val_lbls = self.labels[self.idx_val]
                test_lbls = self.labels[self.idx_test]
                cnt_wait = 0
                best = 1e-9
                output_acc = 1e-9
                totalL = []
                for epoch_i in range(self.args.nb_epochs):
                    env_model.train()
                    optimiser.zero_grad()
                    embeds = env_model(graph_org_torch, features)
                    embeds_preds = torch.argmax(embeds, dim=1)
                    train_embs = embeds[self.idx_train]
                    val_embs = embeds[self.idx_val]
                    test_embs = embeds[self.idx_test]
                    loss = F.cross_entropy(train_embs, train_lbls)
                    loss.backward()
                    totalL.append(loss.item())
                    optimiser.step()
                    ################STA|Eval|###############
                    if epoch_i % 5 == 0 and epoch_i != 0:
                        env_model.eval()
                        embeds = env_model(graph_org_torch, features)
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
                print("\t[Classification] ACC: {:.4f}".format(output_acc))

    def evalue_graph(self):
        pass
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
                embedding_first = self.env_model.get_first_embedding(features).detach()
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
        return Last_embedding, embedding_first

