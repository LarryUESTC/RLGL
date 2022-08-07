import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
from models.Layers import act_layer
from models.embedder import embedder_single
import os
from evaluate import evaluate


def negative_sampling(adj_ori, sample_times):
    sample_list = []
    for j in range(sample_times):
        sample_iter = []
        i = 0
        while True:
            randnum = np.random.randint(0, adj_ori.shape[0])
            if randnum != i:
                sample_iter.append(randnum)
                i = i + 1
            if len(sample_iter) == adj_ori.shape[0]:
                break
        sample_list.append(sample_iter)
    return sample_list


class AvgNeighbor(nn.Module):
    def __init__(self):
        super(AvgNeighbor, self).__init__()

    def forward(self, seq, adj_ori):
        return torch.mm(adj_ori, seq)


class Discriminator_gmi(nn.Module):
    def __init__(self, n_h1, n_h2):
        super(Discriminator_gmi, self).__init__()
        self.f_k = nn.Bilinear(n_h1, n_h2, 1)
        self.act = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h_c, h_pl, sample_list, s_bias1=None, s_bias2=None):
        sc_1 = torch.squeeze(self.f_k(h_pl, h_c), 1)
        sc_1 = self.act(sc_1)
        sc_2_list = []
        for i in range(len(sample_list)):
            h_mi = h_pl[sample_list[i]]
            sc_2_iter = torch.squeeze(self.f_k(h_mi, h_c), 1)
            sc_2_list.append(sc_2_iter)
        sc_2_stack = torch.stack(sc_2_list)
        sc_2 = self.act(sc_2_stack)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        return sc_1, sc_2


class GCN_gmi(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN_gmi, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = act_layer(act)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.mm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out), seq_fts


class gmi(nn.Module):
    def __init__(self, n_in, n_h, activation='prelu'):
        super(gmi, self).__init__()
        self.gcn1 = GCN_gmi(n_in, n_h,
                            activation)  # if on citeseer and pubmed, the encoder is 1-layer GCN, you need to modify it
        self.gcn2 = GCN_gmi(n_h, n_h, activation)
        self.disc1 = Discriminator_gmi(n_in, n_h)
        self.disc2 = Discriminator_gmi(n_h, n_h)
        self.avg_neighbor = AvgNeighbor()
        self.prelu = act_layer(activation)
        self.sigm = nn.Sigmoid()

    def forward(self, seq1, adj_ori, neg_num, adj, samp_bias1, samp_bias2):
        h_1, h_w = self.gcn1(seq1, adj)
        h_2, _ = self.gcn2(h_1, adj)
        h_neighbor = self.prelu(self.avg_neighbor(h_w, adj_ori))
        """FMI (X_i consists of the node i itself and its neighbors)"""
        # I(h_i; x_i)
        res_mi_pos, res_mi_neg = self.disc1(h_2, seq1, negative_sampling(adj_ori, neg_num), samp_bias1,
                                            samp_bias2)
        # I(h_i; x_j) node j is a neighbor
        res_local_pos, res_local_neg = self.disc2(h_neighbor, h_2, negative_sampling(adj_ori, neg_num),
                                                  samp_bias1, samp_bias2)
        """I(w_ij; a_ij)"""
        adj_rebuilt = self.sigm(torch.mm(h_2, torch.t(h_2)))

        return res_mi_pos, res_mi_neg, res_local_pos, res_local_neg, adj_rebuilt

    # detach the return variables
    def embed(self, seq, adj):
        h_1, _ = self.gcn1(seq, adj)
        h_2, _ = self.gcn2(h_1, adj)

        return h_2.detach()


class GMI(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def sp_func(self, arg):
        return torch.log(1 + torch.exp(arg))

    def mi_loss_jsd(self, pos, neg):
        e_pos = torch.mean(self.sp_func(-pos))
        e_neg = torch.mean(torch.mean(self.sp_func(neg), 0))
        return e_pos + e_neg

    def reconstruct_loss(self, pre, gnd):
        nodes_n = gnd.shape[0]
        edges_n = np.sum(gnd) / 2
        weight1 = (nodes_n * nodes_n - edges_n) * 1.0 / edges_n
        weight2 = nodes_n * nodes_n * 1.0 / (nodes_n * nodes_n - edges_n)
        gnd = torch.FloatTensor(gnd).cuda()
        temp1 = gnd * torch.log(pre + (1e-10)) * (-weight1)
        temp2 = (1 - gnd) * torch.log(1 - pre + (1e-10))
        return torch.mean(temp1 - temp2) * weight2

    def training(self):
        features = self.features.to(self.args.device)
        adj = self.adj_list[0].to(self.args.device)
        adj_ori = self.adj_list[2]

        print("Started training...")
        model = gmi(self.args.ft_size, self.args.hid_dim).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)

        cnt_wait = 0
        best = 1e9

        adj_dense = adj_ori.numpy()
        adj_target = adj_dense + np.eye(adj_dense.shape[0])
        adj_row_avg = 1.0 / np.sum(adj_dense, axis=1)
        adj_row_avg[np.isnan(adj_row_avg)] = 0.0
        adj_row_avg[np.isinf(adj_row_avg)] = 0.0
        adj_dense = adj_dense * 1.0
        for i in range(adj_ori.shape[0]):
            adj_dense[i] = adj_dense[i] * adj_row_avg[i]
        adj_ori = torch.from_numpy(adj_dense).to(self.args.device)

        tbar = tqdm(range(self.args.nb_epochs))
        for _ in tbar:
            model.train()
            optimiser.zero_grad()
            res = model(features, adj_ori, self.args.negative_num, adj, None, None)

            loss = self.args.alpha * self.mi_loss_jsd(res[0], res[1]) \
                   + self.args.beta * self.mi_loss_jsd(res[2], res[3]) \
                   + self.args.gamma * self.reconstruct_loss(res[4], adj_target)
            tbar.set_description('Loss:{:.6f}'.format(loss.item()))
            loss.backward()
            optimiser.step()
            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(model.state_dict(), self.args.save_root + '/model.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                break
        model.load_state_dict(torch.load(self.args.save_root + '/model.pkl'))
        print("Evaluating...")
        model.eval()
        embeds = model.embed(features, adj)
        acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, k1, k2, st = evaluate(
            embeds, self.idx_train, self.idx_val, self.idx_test, self.labels,
            seed=self.args.seed, epoch=self.args.test_epo, lr=self.args.test_lr)
        return acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, k1, k2, st
