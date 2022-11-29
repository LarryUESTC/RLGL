import time
from utils import process
from models.embedder import embedder_single
import os
from evaluate import evaluate, accuracy
from models.Layers import SUGRL_Fast, GNN_Model
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import setproctitle
setproctitle.setproctitle('PLLL')
from datetime import datetime

def KDloss_0( P, Q):
    T = 1  #todo
    # P = P.log()
    # Q = F.log_softmax(Q / T, dim=1)
    # P = F.softmax(P / T, dim=1)
    Q = F.softmax(Q / T, dim=1)
    l_kl = F.kl_div(Q, P, size_average=False) * (T ** 2) #/ Q.shape[0]
    return l_kl

def KDloss_1( P, Q):
    T = 1  #todo
    # P = P.log()
    Q = F.log_softmax(Q / T, dim=1)
    # P = F.softmax(P / T, dim=1)
    # Q = F.softmax(Q / T, dim=1)
    l_kl = F.kl_div(Q, P, size_average=False) * (T ** 2) #/ Q.shape[0]
    return l_kl
def KDloss_2( P, Q):
    T = 1  #todo
    P = P.log()
    # Q = F.log_softmax(Q / T, dim=1)
    # P = F.softmax(P / T, dim=1)
    Q = F.softmax(Q / T, dim=1)
    l_kl = F.kl_div(Q, P, size_average=False) * (T ** 2) #/ Q.shape[0]
    return l_kl
def KDloss_3( P, Q):
    T = 1  #todo
    # P = P.log()
    # Q = F.log_softmax(Q / T, dim=1)
    P = F.softmax(P / T, dim=1)
    Q = F.softmax(Q / T, dim=1)
    l_kl = F.kl_div(Q, P, size_average=False) * (T ** 2) #/ Q.shape[0]
    return l_kl
def KDloss_4( P, Q):
    T = 1  #todo
    # P = P.log()
    Q = F.log_softmax(Q / T, dim=1)
    P = F.softmax(P / T, dim=1)
    # Q = F.softmax(Q / T, dim=1)
    l_kl = F.kl_div(Q, P, size_average=False) * (T ** 2) #/ Q.shape[0]
    return l_kl
def KDloss_5( P, Q):
    T = 1  #todo
    # P = P.log()
    Q = F.log_softmax(Q / T, dim=1)
    P = F.log_softmax(P / T, dim=1)
    # Q = F.softmax(Q / T, dim=1)
    l_kl = F.kl_div(Q, P, size_average=False) * (T ** 2) #/ Q.shape[0]
    return l_kl

KD_list = [KDloss_0,KDloss_1,KDloss_2,KDloss_3,KDloss_4,KDloss_5]

class RecordeBuffer:
    def __init__(self):
        self.psudo_labels = []
        self.entropy = []
        self.lenth = 0
    def clear(self):
        del self.psudo_labels[:]
        del self.entropy[:]
    def update(self):
        self.lenth = len(self.psudo_labels)
        if self.lenth > 100:
            del self.psudo_labels[0]
            del self.entropy[0]

def calc_entropy(input_tensor):
    lsm = nn.LogSoftmax()
    log_probs = lsm(input_tensor, dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(-1)
    return entropy

class GCNCR(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.cfg = args.cfg
        # if not os.path.exists(self.args.save_root):
        #     os.makedirs(self.args.save_root)
        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.cfg.append(nb_classes)
        self.model = GNN_Model(self.args.ft_size, cfg=self.cfg, final_mlp = 0, gnn = self.args.gnn, dropout=self.args.random_aug_feature).to(self.args.device)
        self.buffer = RecordeBuffer()
        current_time = datetime.now().strftime('%b%d_%H-%M:%S')
        logdir = os.path.join('runs3', current_time + '_SemiCR_'+ str(self.args.seed))
        self.writer_tb = SummaryWriter(log_dir = logdir)

    def training(self):

        features = self.features.to(self.args.device)
        graph_org = self.dgl_graph.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)
        print("Started training...")


        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay = self.args.wd)
        xent = nn.CrossEntropyLoss()
        train_lbls = self.labels[self.idx_train]
        val_lbls = self.labels[self.idx_val]
        test_lbls = self.labels[self.idx_test]
        train_id = [i.item() for i in self.idx_train]
        idx_unlabel = [j for j in range(len(self.labels)) if train_id.__contains__(j)==False ]
        self.idx_unlabel = torch.from_numpy(np.array(idx_unlabel)).to(self.idx_train.device)
        cnt_wait = 0
        best = 1e-9
        output_acc = 1e-9
        stop_epoch = 0

        start = time.time()
        totalL = []
        # features = F.normalize(features)
        for epoch in range(self.args.nb_epochs):
            self.model.train()
            optimiser.zero_grad()

            # A_a, X_a = process.RA(graph_org.cpu(), features, self.args.random_aug_feature,self.args.random_aug_edge)
            # A_a = A_a.add_self_loop().to(self.args.device)

            embeds = self.model(graph_org_torch, features)
            embeds_preds = torch.argmax(embeds, dim=1)

            self.buffer.psudo_labels.append(F.softmax(embeds.detach(),dim=-1)**2)
            self.buffer.entropy.append(calc_entropy(embeds).detach())
            self.buffer.update()

            train_embs = embeds[self.idx_train]
            val_embs = embeds[self.idx_val]
            test_embs = embeds[self.idx_test]

            loss_ce = F.cross_entropy(train_embs, train_lbls)
            if epoch >= 10:
                loss_psudo = KD_list[3](self.buffer.psudo_labels[-5][self.idx_unlabel], embeds[self.idx_unlabel]) * 0.01
            else:
                loss_psudo = loss_ce*0
            loss = loss_ce + loss_psudo
            self.writer_tb.add_scalar('loss_ce', loss_ce.item(), epoch)
            self.writer_tb.add_scalar('loss_psudo', loss_psudo.item(), epoch)

            loss.backward()
            totalL.append(loss.item())
            optimiser.step()

            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0 :
                self.model.eval()
                # A_a, X_a = process.RA(graph_org.cpu(), features, 0, 0)
                # A_a = A_a.add_self_loop().to(self.args.device)
                embeds = self.model(graph_org_torch, features)
                val_acc = accuracy(embeds[self.idx_val], val_lbls)
                test_acc = accuracy(embeds[self.idx_test], test_lbls)
                self.writer_tb.add_scalar('Test_acc', test_acc, epoch)
                self.writer_tb.add_scalar('val_acc', val_acc, epoch)
                # print(test_acc.item())
                # early stop
                stop_epoch = epoch
                if val_acc > best:
                    best = val_acc
                    output_acc = test_acc.item()
                    cnt_wait = 0
                    # torch.save(model.state_dict(), 'saved_model/best_{}.pkl'.format(self.args.dataset))
                else:
                    cnt_wait += 1
                if cnt_wait == self.args.patience:
                    # print("Early stopped!")
                    break
            ################END|Eval|###############


        training_time = time.time() - start
        # print("training time:{}s".format(training_time))
        # print("Evaluating...")
        print("\t[Classification] ACC: {:.4f} | stop_epoch: {:}| training_time: {:.4f} ".format(
            output_acc, stop_epoch, training_time))
        return output_acc, training_time, stop_epoch