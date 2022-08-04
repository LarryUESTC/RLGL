import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
from models.Layers import GNN_layer, act_layer, AvgReadout,Discriminator
from models.embedder import embedder_single
import os
from utils.process import compute_ppr
from evaluate import evaluate


class mvgrl(nn.Module):
    def __init__(self, n_in, n_h):
        super(mvgrl, self).__init__()
        self.gcn1 = GNN_layer('GCN_org', n_in, n_h)
        self.gcn2 = GNN_layer('GCN_org', n_in, n_h)
        self.activation = act_layer('prelu')
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, diff):
        h_1 = self.activation(self.gcn1(adj, seq1))
        c_1 = self.read(h_1)
        c_1 = self.sigm(c_1)

        h_2 = self.activation(self.gcn2(diff, seq1))
        c_2 = self.read(h_2)
        c_2 = self.sigm(c_2)

        h_3 = self.activation(self.gcn1(adj, seq2))
        h_4 = self.activation(self.gcn2(diff, seq2))

        r_1 = self.disc(c_1,h_2,h_4)
        r_2 = self.disc(c_2,h_1,h_3)
        ret = torch.cat((r_1,r_2))

        return ret, h_1, h_2

    def embed(self, seq, adj, diff):
        h_1 = self.activation(self.gcn1(adj, seq))
        c = self.read(h_1)

        h_2 = self.activation(self.gcn2(diff, seq))
        return (h_1 + h_2).detach(), c.detach()


class MVGRL(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):
        features = self.features.to(self.args.device)
        adj = self.adj_list[0].to(self.args.device)
        diff = compute_ppr(self.adj_list[2]).to(self.args.device)
        print("Started training...")

        lbl_1 = torch.ones(self.args.sample_size, device=self.args.device)
        lbl_2 = torch.zeros(self.args.sample_size, device=self.args.device)
        lbl_3 = torch.ones(self.args.sample_size,device=self.args.device)
        lbl_4 = torch.zeros(self.args.sample_size, device=self.args.device)
        lbl = torch.cat((lbl_1, lbl_2,lbl_3,lbl_4))


        model = mvgrl(self.args.ft_size, self.args.hid_dim).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)

        b_xent = nn.BCEWithLogitsLoss()

        cnt_wait = 0
        best = 1e9
        tbar = tqdm(range(self.args.nb_epochs))

        for _ in tbar:
            idx = np.random.randint(0, adj.shape[-1] - self.args.sample_size + 1)

            ba = adj[idx: idx + self.args.sample_size, idx: idx + self.args.sample_size]
            bd = diff[idx: idx + self.args.sample_size, idx: idx + self.args.sample_size]
            bf = features[idx: idx + self.args.sample_size]

            idx = np.random.permutation(self.args.sample_size)
            shuf_fts = bf[idx, :]
            shuf_fts = shuf_fts.to(self.args.device)

            model.train()
            optimiser.zero_grad()

            logits, __, __ = model(bf, shuf_fts, ba, bd)
            loss = b_xent(logits, lbl)
            loss.backward()
            optimiser.step()
            tbar.set_description('Loss: {:0.4f}'.format(loss.item()))

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
        embeds, __= model.embed(features,adj,diff)
        acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, k1, k2, st = evaluate(
            embeds, self.idx_train, self.idx_val, self.idx_test, self.labels,
            seed=self.args.seed, epoch=self.args.test_epo, lr=self.args.test_lr)
        return acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, k1, k2, st
