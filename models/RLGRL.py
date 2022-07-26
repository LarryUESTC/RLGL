import time
from utils import process
from embedder import embedder_single
import os
from tqdm import tqdm
from evaluate import evaluate, accuracy
from models.Net import SUGRL_Fast, GCN_Fast, Action_Net
import numpy as np
import random as random
import torch.nn.functional as F
import torch
import torch.nn as nn
from models.dqn_agent_pytorch import DQNAgent
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
        self.graph_org_torch = self.adj_list[0].to(self.args.device)
        self.features = self.features.to(self.args.device)
        self.action_num = 5
        self.init_k_hop(self.action_num)
        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.cfg.append(nb_classes)

        self.agent = DQNAgent(scope='dqn',
                         action_num=self.action_num,
                         replay_memory_size=int(1e4),
                         replay_memory_init_size=500,
                         norm_step=200,
                         state_shape=env.observation_space.shape,
                         mlp_layers=[32, 64, 128, 64, 32],
                         device= self.args.device
                         )


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

    def init_k_hop(self, max_hop):

        dd = self.graph_org_torch
        self.adjs = [dd]
        for i in range(max_hop):
            dd = torch.mm(dd, self.graph_org_torch.t())
            self.adjs.append(dd)

    def reset(self):
        index = self.train_indexes[self.i]
        state = self.data.x[index].to('cpu').numpy()
        self.optimizer.zero_grad()
        return state

    def _set_action_space(self, _max):
        self.action_num = _max
        self.action_space = Discrete(_max)

    def _set_observation_space(self, observation):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        self.model.train()
        self.optimizer.zero_grad()
        if self.random == True:
            action = random.randint(1, 5)
        # train one step
        index = self.train_indexes[self.i]
        pred = self.model(action, self.data)[index]
        pred = pred.unsqueeze(0)
        y = self.data.y[index]
        y = y.unsqueeze(0)
        F.nll_loss(pred, y).backward()
        self.optimizer.step()

        # get reward from validation set
        val_acc = self.eval_batch()

        # get next state
        self.i += 1
        self.i = self.i % len(self.train_indexes)
        next_index = self.train_indexes[self.i]
        # next_state = self.data.x[next_index].to('cpu').numpy()
        next_state = self.data.x[next_index].numpy()
        if self.i == 0:
            done = True
        else:
            done = False
        return next_state, val_acc, done, "debug"

    def reset2(self):
        start = self.i
        end = (self.i + self.batch_size) % len(self.train_indexes)
        index = self.train_indexes[start:end]
        state = self.data.x[index].to('cpu').numpy()
        self.optimizer.zero_grad()
        return state

    def step2(self, actions):
        self.model.train()
        self.optimizer.zero_grad()
        start = self.i
        end = (self.i + self.batch_size) % len(self.train_indexes)
        index = self.train_indexes[start:end]
        done = False
        for act, idx in zip(actions, index):
            if self.gcn == True or self.enable_dlayer == False:
                act = self.max_layer
            self.buffers[act].append(idx)
            if len(self.buffers[act]) >= self.batch_size:
                self.train(act, self.buffers[act])
                self.buffers[act] = []
                done = True
        if self.gcn == True or self.enable_skh == False:
            ### Random ###
            self.i += min((self.i + self.batch_size) % self.batch_size, self.batch_size)
            start = self.i
            end = (self.i + self.batch_size) % len(self.train_indexes)
            index = self.train_indexes[start:end]
        else:
            index = self.stochastic_k_hop(actions, index)
        next_state = self.data.x[index].to('cpu').numpy()
        # next_state = self.data.x[index].numpy()
        val_acc_dict = self.eval_batch()
        val_acc = [val_acc_dict[a] for a in actions]
        test_acc = self.test_batch()
        baseline = np.mean(np.array(self.past_performance[-self.baseline_experience:]))
        self.past_performance.extend(val_acc)
        reward = [100 * (each - baseline) for each in val_acc]  # FIXME: Reward Engineering
        r = np.mean(np.array(reward))
        val_acc = np.mean(val_acc)
        return next_state, reward, [done] * self.batch_size, (val_acc, r)

    def stochastic_k_hop(self, actions, index):
        next_batch = []
        for idx, act in zip(index, actions):
            prob = self.adjs[act].getrow(idx).toarray().flatten()
            cand = np.array([i for i in range(len(prob))])
            next_cand = np.random.choice(cand, p=prob)
            next_batch.append(next_cand)
        return next_batch

    def train(self, action, indexes):
        self.model.train()
        pred = self.model(action, self.data)[indexes]
        y = self.data.y[indexes]
        F.nll_loss(pred, y).backward()
        self.optimizer.step()

    def eval_batch(self):
        self.model.eval()
        batch_dict = {}
        val_index = np.where(self.data.val_mask.to('cpu').numpy() == True)[0]
        val_states = self.data.x[val_index].to('cpu').numpy()
        if self.random == True:
            val_acts = np.random.randint(1, 5, len(val_index))
        elif self.gcn == True or self.enable_dlayer == False:
            val_acts = np.full(len(val_index), 3)
        else:
            val_acts = self.policy.eval_step(val_states)
        s_a = zip(val_index, val_acts)
        for i, a in s_a:
            if a not in batch_dict.keys():
                batch_dict[a] = []
            batch_dict[a].append(i)
        # acc = 0.0
        acc = {a: 0.0 for a in range(self.max_layer)}
        for a in batch_dict.keys():
            idx = batch_dict[a]
            logits = self.model(a, self.data)
            pred = logits[idx].max(1)[1]
            # acc += pred.eq(self.data.y[idx]).sum().item() / len(idx)
            acc[a] = pred.eq(self.data.y[idx]).sum().item() / len(idx)
        # acc = acc / len(batch_dict.keys())
        return acc

    def test_batch(self):
        self.model.eval()
        batch_dict = {}
        test_index = np.where(self.data.test_mask.to('cpu').numpy() == True)[0]
        val_states = self.data.x[test_index].to('cpu').numpy()
        if self.random == True:
            val_acts = np.random.randint(1, 5, len(test_index))
        elif self.gcn == True or self.enable_dlayer == False:
            val_acts = np.full(len(test_index), 3)
        else:
            val_acts = self.policy.eval_step(val_states)
        s_a = zip(test_index, val_acts)
        for i, a in s_a:
            if a not in batch_dict.keys():
                batch_dict[a] = []
            batch_dict[a].append(i)
        acc = 0.0
        for a in batch_dict.keys():
            idx = batch_dict[a]
            logits = self.model(a, self.data)
            pred = logits[idx].max(1)[1]
            acc += pred.eq(self.data.y[idx]).sum().item() / len(idx)
        acc = acc / len(batch_dict.keys())
        return acc

    def check(self):
        self.model.eval()
        train_index = np.where(self.data.train_mask.to('cpu').numpy() == True)[0]
        tr_states = self.data.x[train_index].to('cpu').numpy()
        tr_acts = self.policy.eval_step(tr_states)

        val_index = np.where(self.data.val_mask.to('cpu').numpy() == True)[0]
        val_states = self.data.x[val_index].to('cpu').numpy()
        val_acts = self.policy.eval_step(val_states)

        test_index = np.where(self.data.test_mask.to('cpu').numpy() == True)[0]
        test_states = self.data.x[test_index].to('cpu').numpy()
        test_acts = self.policy.eval_step(test_states)

        return (train_index, tr_states, tr_acts), (val_index, val_states, val_acts), (
        test_index, test_states, test_acts)

