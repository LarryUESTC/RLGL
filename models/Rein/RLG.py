import time
from models.embedder import embedder_single
from evaluate import accuracy
from models.Layers import Env_Net_RLG, act_layer, GNN_Model
import numpy as np
import random as random
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from collections import defaultdict

np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

def get_A_r(adj, r):
    adj_label = adj
    for i in range(r - 1):
        adj_label = adj_label @ adj

    return adj_label

def Ncontrast(x_dis, adj_label, tau):
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos +1e-8).mean()
    return loss

def KDloss_0( y, teacher_scores):
    T = 4
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T ** 2) / y.shape[0]
    return l_kl

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

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

class Act_Model(nn.Module):
    def __init__(self, n_in ,cfg = None, batch_norm=True, act='elu', dropout = 0.2, final_mlp = 0):
        super(Act_Model, self).__init__()

        self.dropout = dropout
        self.bat = batch_norm
        self.act = act
        self.final_mlp = final_mlp > 0
        self.cfg = cfg
        self.layer_num = len(cfg)
        in_channels = n_in
        self.norm_0 = Norm(in_channels)
        self.mlp_1 = nn.Linear(in_channels, cfg[0])
        self.norm_1 = Norm(cfg[0])
        self.trans_1 = MultiHeadAttention_new(heads, d_model,d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)


        # for m in self.modules():
        #     self.weights_init(m)
    # def weights_init(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight.data)
    #         if m.bias is not None:
    #             m.bias.data.fill_(0.0)
    #     if isinstance(m, nn.Sequential):
    #         for mm in m:
    #             try:
    #                 torch.nn.init.xavier_uniform_(mm.weight.data)
    #             except:
    #                 pass


    def forward(self, x_input):
        x_input = self.norm_0(x_input)
        x_1 = self.norm_1(F.elu(self.mlp_1(x_input)))
        x_2 = self.trans_1(x_1, x_1, x_1)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # scores = torch.where(scores > 1.1 * scores.mean(), scores, torch.zeros_like(scores))

    if mask is not None:
        mask = mask.unsqueeze(0)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    scores = torch.where(scores > scores.mean(), scores, torch.zeros_like(scores))
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention_new(nn.Module):
    def __init__(self, heads, d_model_in, d_model_out, dropout=0.1):
        super().__init__()

        self.d_model = d_model_out
        self.d_k = d_model_out // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model_in, d_model_out)
        self.v_linear = nn.Linear(d_model_in, d_model_out)
        self.k_linear = nn.Linear(d_model_in, d_model_out)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model_out, d_model_out)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, self.h, self.d_k)
        q = self.q_linear(q).view(bs, self.h, self.d_k)
        v = self.v_linear(v).view(bs, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 0)
        q = q.transpose(1, 0)
        v = v.transpose(1, 0)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(0, 1).contiguous().view(bs, self.d_model)
        output = self.out(concat)

        return output


# class gcn_env(object):
#     def __init__(self, args, adj, feature, data_split, label, lr=0.01, weight_decay=5e-4,  batch_size=128, policy=""):
#         self.device = args.device
#         self.args = args
#         self.adj = adj
#         self.feature = feature.to(self.device)
#
#         self.model = Env_Net_RLG(self.args.ft_size, self.args.cfg).to(self.device)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
#         self.idx_train, self.idx_val, self.idx_test = zip(*data_split)
#         self.train_indexes = np.where(np.array(self.idx_train) == True)[0]
#         self.val_indexes = np.where(np.array(self.idx_val) == True)[0]
#         self.test_indexes = np.where(np.array(self.idx_test) == True)[0]
#         self.label = label
#         self.batch_size = len(self.train_indexes) - 1
#         self.i = 0
#         self.val_acc = 0.0
#         self.policy = policy
#
#         self.buffers = defaultdict(list)
#
#         self.past_acc = []
#         self.past_loss = []
#
#     def training(self):
#
#         features = self.feature.to(self.args.device)
#         graph_org = self.adj.to(self.args.device)
#
#         print("Started training...")
#
#         optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
#         xent = nn.CrossEntropyLoss()
#         train_lbls = self.label[self.idx_train]
#         val_lbls = self.label[self.idx_val]
#         test_lbls = self.label[self.idx_test]
#
#         cnt_wait = 0
#         best = 1e-9
#         output_acc = 1e-9
#         stop_epoch = 0
#
#         start = time.time()
#         totalL = []
#         # features = F.normalize(features)
#         for epoch in range(self.args.nb_epochs):
#             self.model.train()
#             optimiser.zero_grad()
#
#             embeds = self.model(graph_org, features)
#             embeds_preds = torch.argmax(embeds, dim=1)
#
#             train_embs = embeds[self.idx_train]
#             val_embs = embeds[self.idx_val]
#             test_embs = embeds[self.idx_test]
#
#             loss = F.cross_entropy(train_embs, train_lbls)
#
#             loss.backward()
#             totalL.append(loss.item())
#             optimiser.step()
#
#             ################STA|Eval|###############
#             if epoch % 5 == 0 and epoch != 0:
#                 self.model.eval()
#                 # A_a, X_a = process.RA(graph_org.cpu(), features, 0, 0)
#                 # A_a = A_a.add_self_loop().to(self.args.device)
#                 embeds = self.model(graph_org, features)
#                 train_acc = accuracy(embeds[self.idx_train], train_lbls)
#                 val_acc = accuracy(embeds[self.idx_val], val_lbls)
#                 test_acc = accuracy(embeds[self.idx_test], test_lbls)
#                 # print(test_acc.item())
#                 # early stop
#                 stop_epoch = epoch
#                 if val_acc > best:
#                     best = val_acc
#                     output_acc = test_acc.item()
#                     cnt_wait = 0
#                     # torch.save(model.state_dict(), 'saved_model/best_{}.pkl'.format(self.args.dataset))
#                 else:
#                     cnt_wait += 1
#                 if cnt_wait == self.args.patience:
#                     # print("Early stopped!")
#                     break
#             ################END|Eval|###############
#
#         training_time = time.time() - start
#         self.past_acc.append(train_acc.item())
#         self.past_loss.append(loss.item())
#
#         return output_acc, training_time, stop_epoch
#
#     def evaluing(self, new_graph):
#         self.model.eval()
#         ################STA|Eval|###############
#         train_lbls = self.label[self.idx_train]
#         val_lbls = self.label[self.idx_val]
#         test_lbls = self.label[self.idx_test]
#         self.model.eval()
#         # A_a, X_a = process.RA(graph_org.cpu(), features, 0, 0)
#         # A_a = A_a.add_self_loop().to(self.args.device)
#         embeds = self.model(new_graph, self.feature)
#         train_embs = embeds[self.idx_train]
#         val_embs = embeds[self.idx_val]
#         test_embs = embeds[self.idx_test]
#
#         loss = F.cross_entropy(train_embs, train_lbls)
#         train_acc = accuracy(train_embs, train_lbls)
#         val_acc = accuracy(val_embs, val_lbls)
#         test_acc = accuracy(test_embs, test_lbls)
#
#         return loss
#
#     def reset(self):
#         index = self.train_indexes[self.i]
#         state = self.feature[index].to('cpu').numpy()
#         self.optimizer.zero_grad()
#         return state
#
#
#     def step(self, action):
#         self.model.train()
#         self.optimizer.zero_grad()
#         if self.random == True:
#             action = random.randint(1, 5)
#         # train one step
#         index = self.train_indexes[self.i]
#         pred = self.model(action, self.feature, self.adj)[index]
#         pred = pred.unsqueeze(0)
#         y = self.label[index]
#         y = y.unsqueeze(0)
#         F.nll_loss(pred, y).backward()
#         self.optimizer.step()
#
#         # get reward from validation set
#         val_acc = self.eval_batch()
#
#         # get next state
#         self.i += 1
#         self.i = self.i % len(self.train_indexes)
#         next_index = self.train_indexes[self.i]
#         # next_state = self.data.x[next_index].to('cpu').numpy()
#         next_state = self.feature[next_index].numpy()
#         if self.i == 0:
#             done = True
#         else:
#             done = False
#         return next_state, val_acc, done, "debug"
#
#     def reset2(self):
#         start = self.i
#         end = (self.i + self.batch_size) % len(self.train_indexes)
#         index = self.train_indexes[start:end]
#         state = self.feature[index].to('cpu').numpy()
#         self.optimizer.zero_grad()
#         return state
#
#     def step2(self, actions):
#         self.model.train()
#         self.optimizer.zero_grad()
#         start = self.i
#         end = (self.i + self.batch_size) % len(self.train_indexes)
#         index = self.train_indexes[start:end]
#         done = False
#         for act, idx in zip(actions, index):
#             if self.gcn == True or self.enable_dlayer == False:
#                 act = self.max_layer
#             self.buffers[act].append(idx)
#             if len(self.buffers[act]) >= self.batch_size:
#                 self.train(act, self.buffers[act])
#                 self.buffers[act] = []
#                 done = True
#         if self.gcn == True or self.enable_skh == False:
#             ### Random ###
#             self.i += min((self.i + self.batch_size) % self.batch_size, self.batch_size)
#             start = self.i
#             end = (self.i + self.batch_size) % len(self.train_indexes)
#             index = self.train_indexes[start:end]
#         else:
#             index = self.stochastic_k_hop(actions, index)
#         next_state = self.feature[index].to('cpu').numpy()
#         # next_state = self.data.x[index].numpy()
#         val_acc_dict = self.eval_batch()
#         val_acc = [val_acc_dict[a] for a in actions]
#         test_acc = self.test_batch()
#         baseline = np.mean(np.array(self.past_performance[-self.baseline_experience:]))
#         self.past_performance.extend(val_acc)
#         reward = [100 * (each - baseline) for each in val_acc]  # FIXME: Reward Engineering
#         r = np.mean(np.array(reward))
#         val_acc = np.mean(val_acc)
#         return next_state, reward, [done] * self.batch_size, (val_acc, r)

class RLG(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.cfg = args.cfg
        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.cfg.append(nb_classes)
        self.graph_org_torch = self.adj_list[0]

        self.action_num = 5
        self.max_episodes = 325
        self.max_timesteps = 10

        self.data_split = zip(self.idx_train.cpu(), self.idx_val.cpu(), self.idx_test.cpu())
        self.env_model = GNN_Model(self.args.ft_size, cfg=self.cfg[1:], final_mlp = 0, gnn = self.args.gnn, dropout=self.args.random_aug_feature).to(self.args.device)
        self.act_model = MLP_Model(self.args.ft_size, cfg=self.cfg, final_mlp = 0, dropout=self.args.random_aug_feature).to(self.args.device)

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

    def act_training(self, env_embedding):
        features = self.features.to(self.args.device)
        graph_org = self.dgl_graph.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)
        print("Started training...")
        idx_test_val = self.idx_test + self.idx_val
        batch_size = 64

        for epoch in range(self.args.nb_epochs):
            batch_idx = random.sample(idx_test_val, batch_size)
            batch_input = features(batch_idx)



        return None

    def training(self):
        env_embedding = self.pre_training()
        self.act_training(env_embedding)
        return None

