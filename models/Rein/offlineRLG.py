import torch
import time
import numpy as np
import copy
from models.embedder import embedder_single
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from models.Layers import act_layer, GNN_Model, GNN_pre_Model
from models.NodeClas.Semi_SELFCONS import SELFCONS
from models.Rein.RLG import KDloss_0
from evaluate import accuracy
import setproctitle
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
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

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def get_A_r(adj, r):
    adj_label = adj
    for i in range(r - 1):
        adj_label = adj_label @ adj

    return adj_label

def calc_entropy(input_tensor):
    lsm = nn.LogSoftmax()
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(dim = 1)
    return entropy

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
        self.fc = nn.ModuleList([])
        self.fc += nn.Sequential(
            nn.Linear(input_features * 3, layers[0], bias=bias),
            act_layer(act),
            Norm(layers[0])
        )
        for i in range(len(layers) - 1):
            self.fc += [
                nn.Sequential(
                    nn.Linear(layers[i], layers[i + 1], bias=bias),
                    act_layer(act),
                    Norm(layers[i + 1])
                )
            ]
        self.fc = nn.Sequential(*self.fc)

        # get the lk in every node k. lk is a score, so the output is 1, state-value function
        self.get_lk = nn.Sequential(
            nn.Linear(layers[-1], layers[-1], bias=bias),
            act_layer(act),
            Norm(layers[-1]),
            nn.Linear(layers[-1], 1, bias=bias)
        )
        # get the action value. action ={0,1}, so the output is 2
        self.get_action = nn.Sequential(
            nn.Linear(layers[-1], layers[-1], bias=bias),
            act_layer(act),
            Norm(layers[-1]),
            nn.Linear(layers[-1], 2, bias=bias)
        )
        # use to calculate the embedding of node v

        self.buffer = RecordeBuffer()
        self.candidate_index = None
        self.candidate_baseline = None
        self.candidate_index_reverse = None
        self.A_G = None
        self.A_G_last = None
        self.adj_label_list = []
        self.node_num = 0
        self.pre_embedding_dis = None
        self.env = None
        self.A_N_I =None
        self.A_N  = None
        self.A = None
        self.I = None
        self.reward_matrix = None

    def pi(self, s):
        z = self.fc(s)
        z = self.get_action(z)
        prob = F.softmax(z, dim=-1)
        return prob

    def v(self, s):
        #todo
        s= torch.cat([s[:, :self.input_features * 2], s[:, self.input_features * 2:]*0], dim=-1)
        z = self.fc(s)
        v = self.get_lk(z)
        return v

    def get_reward_0(self,  consider_idx, at, adj_id):
        if adj_id >= 3:
            this_adj = self.pre_embedding_dis
            # avg = this_adj.mean(dim = 1)
            avg = self.candidate_baseline *0.9
            reward = (this_adj[[*range(0, len(consider_idx))], consider_idx] - avg) * (at * 2 - 1)
        else:
            this_adj = (self.adj_label_list[-1] > 0) + 0
            avg = this_adj.sum(dim=1) / (this_adj > 0).sum(dim=1)
            reward = (this_adj[[*range(0, len(consider_idx))], consider_idx] - avg*0.2) * (at*2-1)
        return reward


    def get_reward_1(self,  consider_idx, at):
        reward = (self.pre_embedding_dis[[*range(0, len(consider_idx))], consider_idx] - self.pre_embedding_dis.mean(dim = 1)) \
                 * (at*2-1)
        return reward

    def get_reward_2(self,  consider_idx, at, adj_label):
        reward = (adj_label[[*range(0, len(consider_idx))], consider_idx]  - adj_label.mean(dim = 1))* (at*2-1)
        # reward +=
        return reward*100

    def get_reward_3(self,  consider_idx, at):
        reward = self.reward_matrix[[*range(0, len(consider_idx))], consider_idx,at]
        return reward

    def _init(self, node_num, pre_embedding, input_embedding, adj_label_list, env, graph_org_NI, graph_org_N, graph_org):
        self.node_num = node_num
        self.pre_embedding = pre_embedding
        self.input_embedding = input_embedding
        self.adj_label_list = adj_label_list
        self.env = env
        self.A_N_I= graph_org_NI
        self.A_N =graph_org_N
        self.A = graph_org
        self.I = torch.eye(self.node_num).to(self.env.args.device)
        self._init_nei_group()
        self._reset()
        current_time = datetime.now().strftime('%b%d_%H-%M:%S')
        logdir = os.path.join('runs4', current_time + '_offlineRLG_'+ str(self.env.args.seed))
        self.writer_tb = SummaryWriter(log_dir = logdir)
    def _init_nei_group(self):
        # pre_embedding = F.softmax(self.pre_embedding)
        pre_embedding = self.pre_embedding
        pre_embedding_dis = get_feature_dis(pre_embedding)
        self.pre_embedding_dis = pre_embedding_dis
        topk = 500

        value, candidate_index = torch.topk(pre_embedding_dis, k=topk, dim=-1, largest=True)
        self.candidate_index = candidate_index.tolist()
        self.candidate_baseline = value[:,-1]
        value, candidate_index = torch.topk(pre_embedding_dis, k=topk, dim=-1, largest=False)
        self.candidate_index_reverse = candidate_index.tolist()

    def _sampleNeigh(self):
        sample_list = []
        for i in range(self.node_num):
            index_TABLE = self.candidate_index[i] #+ self.candidate_index_reverse[i]
            sample_index = i
            while sample_index == i:  # 不让它取自己
                sample_index = random.choice(index_TABLE)
            sample_list.append(sample_index)
        self.sample_list = sample_list  # 记录下来 因为后续在改变邻居是要用到
        # sample_list = random.sample(list(range(0, self.node_num)), self.node_num)
        return sample_list

    def _reset(self):
        if self.A_G is not None:
            self.A_G_last = copy.deepcopy(self.A_G)
        self.A_G = torch.eye(self.node_num).to(self.env.args.device)

    def forward(self, epoch):

        embeddings = copy.deepcopy(self.input_embedding)  # change the feature dimension to embedding dimension

        A_G_N = F.normalize(self.A_G, p=1)
        embeddings_neibor = torch.mm(A_G_N, embeddings)
        #todo select self or neibor?
        # batch_size = features.size()[0]
        # self.consider_idx = random.sample(list(range(0, self.node_num)), self.node_num)
        self.consider_idx = self._sampleNeigh()

        # train_index = torch.from_numpy(np.array(range(0, batch_size)))[train_index.tolist()]2
        # consider_idx = random.choices(train_index.tolist(), k=batch_size)
        embeddings_consider = embeddings[self.consider_idx]

        s = torch.cat([embeddings, embeddings_neibor, embeddings_consider], dim=-1)

        T_horizon = 50
        ac_acc_list = []
        if epoch % 5 == 0:
            print(f"\n Epoch {epoch}", end="")
        for t in range(T_horizon):
            z = self.fc(s)
            lk = self.get_action(z)
            at_distribution = F.softmax(lk, dim=-1)
            at_dist = Categorical(at_distribution)
            at = at_dist.sample()

            self.A_G[[*range(0, self.node_num)], self.consider_idx] = at+0.0
            A_G_N = F.normalize(self.A_G, p=1)
            # rt = self.get_reward_0(self.consider_idx, at, adj_id= 2)\
            rt = self.get_reward_3(self.consider_idx, at)
            at_acc = 100 *  ((self.env.labels == self.env.labels[self.consider_idx]) * at).sum() / at.sum()
            ac_acc_list.append(at_acc)
            if epoch%5==0:
                print("A-{}: {:.2f}/{}|".format(t, at_acc, at.sum()), end="")
            if t <5:
                self.writer_tb.add_scalar('global_acc_'+str(t), at_acc, epoch)
                self.writer_tb.add_scalar('global_num_' + str(t), at.sum(), epoch)
                self.writer_tb.add_scalar('reward_' + str(t), rt.detach().mean().item(), epoch)

            embeddings_neibor_next = torch.mm(A_G_N, embeddings)
            # todo select self or neibor?
            # consider_idx_next = random.sample(list(range(0, batch_size)), batch_size)
            self.consider_idx_next = self._sampleNeigh()
            # consider_idx_next = random.choices(train_index.tolist(), k=batch_size)
            embeddings_consider_next = embeddings[self.consider_idx_next]
            s_next = torch.cat([embeddings, embeddings_neibor_next, embeddings_consider_next], dim=1)

            self.buffer.actions.append(at.detach())
            self.buffer.states.append(s.detach())
            self.buffer.states_next.append(s_next.detach())
            self.buffer.logprobs.append(at_dist.log_prob(at).detach())
            self.buffer.rewards.append(rt.detach())
            self.buffer.is_end.append(False)

            self.consider_idx = self.consider_idx_next
            # embeddings_consider = embeddings_consider_next
            s = s_next
        return

    def evlue(self):
        embeddings = copy.deepcopy(self.input_embedding)  # change the feature dimension to embedding dimension
        A_G_2 = F.normalize(self.A, p=1)
        A_G_out = copy.deepcopy(self.A)
        embeddings_neibor = torch.mm(A_G_2, embeddings)


        for consider_idx in range(self.node_num): #range(batch_size, features.size()[0]):
            # num_idx = adj_env.sum(-1)[consider_idx]
            end_neibor = 10

            embeddings_i = embeddings[consider_idx].repeat(self.node_num, 1)
            embeddings_neibor_i = embeddings_neibor[consider_idx].repeat(self.node_num, 1)
            embeddings_consider_i = embeddings #[train_index]
            s = torch.cat([embeddings_i, embeddings_neibor_i, embeddings_consider_i], dim=-1)
            # s_v = torch.cat([embeddings_i, (adj_env.sum(-1)[consider_idx] * embeddings_neibor_i + embeddings_consider_i) / (1 +adj_env.sum(-1)[consider_idx]), embeddings_consider_i], dim=-1)
            # v = self.v(s_v)
            z = self.fc(s)
            lk = self.get_action(z)
            at_distribution = F.softmax(lk, dim=-1)

            at = at_distribution[:, 1].topk(k=end_neibor, dim=0, largest=True, sorted=True)[1]
            A_G_out[consider_idx, at] = 1.0
            # adj_env_out[at, consider_idx] = 1.0

                # max_idx = at_distribution[:, 1].topk(k=20, dim=0, largest=True, sorted=True)[1]
                # at = max_idx[v[max_idx].topk(k=int(5 - num_idx.item() + 1), dim=0, largest=True, sorted=True)[1]].squeeze()
                # adj_env[consider_idx, at] = 1.0
                # adj_env[at, consider_idx] = 1.0

        # adj_env_N = F.normalize(adj_env, p=1)

        return A_G_out


class PPO_Module(nn.Module):
    def __init__(self, input_features, layers, embedding_features, env, n_classes=7, lr=0.0005, gamma=0.99, gae_lambda=0.95,
                 K_epoch=40, eps_clip=0.2, act='relu', device='cpu'):
        super(PPO_Module, self).__init__()

        layers_sa = [128,64,32]
        self.State = nn.Sequential(
            nn.Linear(input_features * 2, layers_sa[0]),
        )
        self.Action = nn.Sequential(
            nn.Linear(1, layers_sa[0]//4),
        )
        self.Reward = nn.Sequential(
            nn.Linear(layers_sa[0] + layers_sa[0]//4, layers_sa[1]),
            act_layer('relu'),
            Norm(layers_sa[1]),
            nn.Linear(layers_sa[1], layers_sa[2]),
            act_layer('relu'),
            Norm(layers_sa[2]),
            # nn.Linear(layers_sa[2], layers_sa[2]),
            # act_layer('relu'),
            # Norm(layers_sa[2]),
            nn.Linear(layers_sa[2], 1),
        )

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
        self.env = env
        self.psudo_labels = None
        self.top_entropy_idx = None
        self.entropy = None
        self.input_embedding = None
        self.A = None
        self.N = None


    def make_batch(self):
        a = torch.stack(self.policy.buffer.actions)
        s = torch.stack(self.policy.buffer.states)
        s_prime = torch.stack(self.policy.buffer.states_next)
        prob_a = torch.stack(self.policy.buffer.logprobs)
        r = torch.stack(self.policy.buffer.rewards)
        # done_mask = torch.cat(self.policy.buffer.is_end)
        self.policy.buffer.clear()

        return a, s, s_prime, r, prob_a

    def update(self, epoch):
        if epoch% 1 == 0:
            self.policy._reset()
        self.policy(epoch)  # get reward state and caction (in polic.buffer)
        a, s, s_prime, r, prob_a = self.make_batch()
        learning_rate = 0.0005
        gamma = 0.98
        lmbda = 0.95
        eps_clip = 0.1
        K_epoch = 5
        T_horizon = s.size()[0]
        batch_size = s.size()[1]
        self.train()
        self.policy.train()
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

    def forward(self):
        # adj_new = self.policy.evlue()
        adj_new = self.policy.A_G_last
        return adj_new

    def init_SA(self, pre_embedding, pre_embedding_entropy, input_embedding, graph_org):
        entropy_top = 1000
        self.input_embedding = input_embedding
        self.A = graph_org
        self.N = self.A.size()[0]
        top_entropy_idx = torch.topk(pre_embedding_entropy, k=entropy_top, dim=-1, largest=False)[1]
        print(accuracy(pre_embedding[top_entropy_idx], self.env.labels[top_entropy_idx]))
        self.idx_train = self.env.idx_train.cpu().numpy().tolist()
        self.top_entropy_idx = []
        for i in top_entropy_idx:
            if i not in self.env.idx_train:
                self.top_entropy_idx.append(i.item())

        self.entropy = pre_embedding_entropy
        self.psudo_labels = pre_embedding.max(1)[1].type_as(self.env.labels)
        self.degree = graph_org.sum(dim = -1)
        self.top_degree_idx = torch.topk(self.degree, k=100, dim=-1, largest=True)[1].cpu().numpy().tolist()

    def make_sa_batch(self, batch_size = 512):
        x = []
        a = []
        y = []
        anchor_list = []
        neibor_list = []
        action_list = [-1.0,1.0]
        consider_list = self.idx_train  + self.top_entropy_idx
        for i in range(batch_size):
            sample_index = random.sample(consider_list, 2)
            anchor = sample_index[0]
            neibor = sample_index[1]
            action = random.sample(action_list, 1)[0]
            reward = 0
            if anchor in self.idx_train:
                anchor_label = self.env.labels[anchor]
                anchor_confidence = 1
            else:
                anchor_label = self.psudo_labels[anchor]
                anchor_confidence = 0.9

            if neibor in self.idx_train:
                neibor_label = self.env.labels[neibor]
                neibor_confidence = 1
            else:
                neibor_label = self.psudo_labels[neibor]
                neibor_confidence = 0.9

            if anchor_label.eq(neibor_label):
                if action == 1.0:
                    reward += 1 * anchor_confidence * neibor_confidence
                else:
                    reward -= 0.25 * anchor_confidence * neibor_confidence
            else:
                if action == 1.0:
                    reward -= 0.5 * anchor_confidence * neibor_confidence
                else:
                    reward += 0.125 * anchor_confidence * neibor_confidence

            anchor_list.append(anchor)
            neibor_list.append(neibor)
            x.append(torch.cat([self.input_embedding[anchor],self.input_embedding[neibor]], dim=-1))
            a.append(action)
            y.append(reward)

        x = torch.stack(x, dim=0)
        a = torch.tensor(a).to(x.device).unsqueeze(dim = -1)
        y = torch.tensor(y).to(x.device).unsqueeze(dim = -1)
        return x, a, y, anchor_list, neibor_list

    def train_SA(self):
        optimizer_SA = torch.optim.Adam(self.parameters(), lr=0.0001)
        for epoch_j in range(500):
            self.train()
            x, a, y, anchor_list, neibor_list = self.make_sa_batch()
            # x = F.dropout(x, 0.1, training=self.training)
            z_0 = self.State(x)
            z_1 = self.Action(a)
            r_input = torch.cat([z_0, z_1], dim=1)
            # r_input = F.dropout(r_input, 0.1, training=self.training)
            r = self.Reward(r_input)
            loss = self.mse_loss(r, y)
            optimizer_SA.zero_grad()
            loss.backward()
            print("Epoch{}- loss: {:.2f}".format(epoch_j, loss.item()))
            optimizer_SA.step()
        self.evalue_SA()

    def evalue_SA(self):
        self.eval()
        reward_matrix = []
        for i in range(self.N):
            anchor = self.input_embedding[i].repeat(self.N, 1)
            neibor = self.input_embedding
            x = torch.cat([anchor, neibor], dim=-1)

            a = torch.tensor([-1.0]).repeat(self.N, 1).to(x.device)
            z_0 = self.State(x)
            z_1 = self.Action(a)
            r_input = torch.cat([z_0, z_1], dim=1)
            r_0 = self.Reward(r_input)

            a = torch.tensor([1.0]).repeat(self.N, 1).to(x.device)
            z_0 = self.State(x)
            z_1 = self.Action(a)
            r_input = torch.cat([z_0, z_1], dim=1)
            r_1 = self.Reward(r_input)

            r_output = torch.cat([r_0, r_1], dim=1)
            reward_matrix.append(r_output.detach())

        self.policy.reward_matrix = torch.stack(reward_matrix, dim=0)


class offlineRLG(embedder_single):
    def __init__(self, args):

        super(offlineRLG, self).__init__(args)
        self.args = args
        self.fake_labels = None
        self.nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.input_dim = 128
        self.model = PPO_Module(self.input_dim, self.args.cfg, self.args.feature_dimension, self, act='gelu',
                                n_classes=self.nb_classes, device=self.args.device).to(self.args.device)

        self.env_model = GNN_pre_Model(self.args.ft_size, cfg=[self.input_dim, 16, self.nb_classes], final_mlp = 0, gnn = self.args.gnn, dropout=self.args.random_aug_feature).to(self.args.device)

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

        pre_embedding, input_embedding = self.pre_training()
        pre_embedding_entropy = calc_entropy(pre_embedding)
        adj_label_list = []
        # adj_label_list.append(graph_org_NI)
        for d in range(1,5):
            adj_label = get_A_r(graph_org_NI, d)
            adj_label_list.append(adj_label)
        node_num = pre_embedding.size()[0]

        self.model.policy._init(node_num, pre_embedding, input_embedding, adj_label_list, self, graph_org_NI, graph_org_N, graph_org)
        self.model.init_SA(pre_embedding, pre_embedding_entropy, input_embedding, graph_org)

        self.model.train_SA()
        test_acc_out = 1e-9
        end_epoch = 0
        for epoch in range(self.args.nb_epochs):
            self.model.train()
            reward = self.model.update(epoch)
            rewards.append(reward)

            if epoch % 50 == 0 and epoch != 0:
                self.model.eval()
                adj_new = self.model()
                graph_org_torch = F.normalize(adj_new +self.adj_list[0].to(self.args.device) , p= 1) #+
                # graph_org_torch = F.normalize(adj_new, p=1) + torch.eye(adj_new.size()[0]).to(adj_new.device)
                test_acc = self.evalue_graph_selfcons(graph_org_torch)
                self.model.policy.writer_tb.add_scalar('Eva_acc', test_acc, epoch)
                if test_acc_out < test_acc:
                    test_acc_out = test_acc
                    end_epoch = epoch

        return test_acc_out, 0, end_epoch

    def evalue_graph(self, graph_org_torch):

        # from utils import process
        # graph_org_torch = process.torch2dgl(graph_org_torch).to(self.args.device)
        env_model = GNN_Model(self.args.ft_size, cfg=[16,  self.nb_classes], final_mlp=0, gnn='GCN_org',
                              dropout=self.args.random_aug_feature).to(self.args.device)
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
        stop_epoch =0
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
            if epoch_i % 25 == 0 and epoch_i != 0:
                env_model.eval()
                embeds = env_model(graph_org_torch, features)
                val_acc = accuracy(embeds[self.idx_val], val_lbls)
                test_acc = accuracy(embeds[self.idx_test], test_lbls)
                stop_epoch = epoch_i
                if val_acc > best:
                    best = val_acc
                    output_acc = test_acc.item()
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if cnt_wait == self.args.patience:
                    break
        print("")
        print("\t \n [Classification] ACC: {:.4f} | stop_epoch: {:} ".format(output_acc, stop_epoch))
        return output_acc

    def evalue_graph_selfcons(self, graph_generate):
        from params import parse_args
        args_evalue = parse_args('Semi', 'SelfCons', self.args.dataset)
        args_evalue.device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
        embedder = SELFCONS(copy.deepcopy(args_evalue))
        embedder.adj_list[0] = graph_generate
        test_acc, training_time, stop_epoch = embedder.training()
        print("\t \n [Classification] ACC: {:.4f} | stop_epoch: {:} ".format(test_acc, stop_epoch))
        return test_acc

    def pre_training(self):

        features = self.features.to(self.args.device)
        # graph_org = self.dgl_graph.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)
        graph_org_torch = F.normalize(self.adj_list[-1].to(self.args.device) + torch.eye(self.adj_list[-1].size()[0]).to(self.args.device), p =1)
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

