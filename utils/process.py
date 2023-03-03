import torch
import numpy as np
import pickle as pkl
import scipy.io as sio
import scipy.sparse as sp
from torch import Tensor
from torch.utils.data import DataLoader, sampler
from scipy.sparse import coo_matrix, diags, csr_matrix
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
from scipy.linalg import fractional_matrix_power, inv
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
import os.path as osp
from torch_geometric.datasets import Planetoid, Amazon
from torchvision.datasets import CIFAR10
import dgl
from sklearn.model_selection import StratifiedShuffleSplit


def load_image_dataset(args=None):
    train_dl = val_dl = test_dl = None
    if args.dataset in ['CIFAR10']:
        class ChunkSampler(sampler.Sampler):
            def __init__(self, num_samples, start=0):
                self.num_samples = num_samples
                self.start = start

            def __iter__(self):
                return iter(range(self.start, self.start + self.num_samples))

            def __len__(self):
                return self.num_samples

        NUM_TRAIN = 50000
        NUM_VAL = 0
        train = CIFAR10('./utils/data', train=True, download=True, transform=args.transform['train'])
        test = CIFAR10('./utils/data', train=False, download=True, transform=args.transform['test'])

        train_dl = DataLoader(train, batch_size=args.batch_size, num_workers=args.workers,
                              sampler=ChunkSampler(NUM_TRAIN))
        val_dl = DataLoader(train, batch_size=args.batch_size, num_workers=args.workers,
                            sampler=ChunkSampler(NUM_VAL, start=NUM_TRAIN))
        test_dl = DataLoader(test, batch_size=args.batch_size, num_workers=args.workers)
    else:
        print(f"Do not find {args.dataset} dataset!")
    return train_dl, val_dl, test_dl


def load_single_graph(args=None, train_ratio=0.1, val_ratio=0.1):
    if args.dataset in ['CiteSeer', 'PubMed', 'Photo', 'Computers']:
        if args.dataset in ['CiteSeer', 'PubMed']:
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/')
            dataset = Planetoid(path, args.dataset)
        elif args.dataset in ['Photo', 'Computers']:
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/')
            dataset = Amazon(path, args.dataset, pre_transform=None)  # transform=T.ToSparseTensor(),
        data = dataset[0]
        # if args.dataset == 'Cora':
        #     idx_train = data.train_mask
        #     idx_val = data.val_mask
        #     idx_val[:] = False
        #     idx_val[range(200, 500)] = True
        #     idx_test = data.test_mask
        #     idx_test[:] = False
        #     idx_test[range(500, 1500)] = True
        # else:
        #     idx_train = data.train_mask
        #     idx_val = data.val_mask
        #     idx_test = data.test_mask
        idx_train = data.train_mask
        idx_val = data.val_mask
        idx_test = data.test_mask
        i = torch.Tensor.long(data.edge_index)
        v = torch.FloatTensor(torch.ones([data.num_edges]))
        A_sp = torch.sparse.FloatTensor(i, v, torch.Size([data.num_nodes, data.num_nodes]))
        A = A_sp.to_dense()
        A_nomal = row_normalize(A)
        I = torch.eye(A.shape[1]).to(A.device)
        A_I = A + I
        A_I_nomal = row_normalize(A_I)
        label = data.y
    elif args.dataset in ['Cora']:
        idx_features_labels = np.genfromtxt("{}{}.content".format("./utils/data/Cora/","cora"), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        def normalize_features(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format("./utils/data/Cora/","cora"), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
            edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize_features(features)
        A_I_nomal = normalize_adj(adj + sp.eye(adj.shape[0]))
        A_nomal = normalize_adj(adj)
        A =  adj.todense()

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        A_I_nomal = torch.FloatTensor(np.array(A_I_nomal.todense()))
        A_nomal = torch.FloatTensor(np.array(A_nomal.todense()))
        A = torch.FloatTensor(np.array(A))

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        return [A_I_nomal, A_nomal, A], features, labels, idx_train, idx_val, idx_test
    else:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/')
        dataset = PygNodePropPredDataset(name=args.dataset, root=path)
        # split_idx = dataset.get_idx_split()
        if args.dataset in ['ogbn-arxiv', 'ogbn-proteins']:
            data = dataset[0]
            data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        elif args.dataset in ['ogbn-mag', 'ogbn-products']:
            if args.dataset in ['ogbn-mag']:
                rel_data = dataset[0]
                # We are only interested in paper <-> paper relations.
                data = Data(
                    x=rel_data.x_dict['paper'],
                    edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
                    y=rel_data.y_dict['paper'])
                data.edge_index = to_undirected(data.edge_index, data.num_nodes)
            else:
                rel_data = dataset[0]
                data = Data(
                    x=rel_data.x,
                    edge_index=rel_data.edge_index,
                    y=rel_data.y)
        A = coo_matrix(
            (np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
            shape=(data.num_nodes, data.num_nodes))
        nb_nodes = data.num_nodes
        I = coo_matrix((np.ones(nb_nodes), (np.arange(0, nb_nodes, 1), np.arange(0, nb_nodes, 1))),
                       shape=(nb_nodes, nb_nodes))
        A_nomal = row_normalize_sparse(A)
        A_I = A + I  # coo_matrix(sp.eye(adj.shape[0]))
        A_I_nomal = row_normalize_sparse(A_I)
        label = data.y
        A_I_nomal = sparse_mx_to_torch_sparse_tensor(A_I_nomal)
        A_nomal = sparse_mx_to_torch_sparse_tensor(A_nomal)
        A = sparse_mx_to_torch_sparse_tensor(A)

        random_split = torch.randperm(nb_nodes)
        train_num = int(nb_nodes * train_ratio)
        val_num = int(nb_nodes * val_ratio)
        idx_train = random_split[:train_num]
        idx_val = random_split[train_num:train_num + val_num]
        idx_test = random_split[train_num + val_num:]

    return [A_I_nomal, A_nomal, A], data.x, label, idx_train, idx_val, idx_test


def load_cora(path="./data/Cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    def normalize_features(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_acm_mat(sc=3):
    data = sio.loadmat('utils/data/acm.mat')
    label = data['label']

    adj_edge1 = data["PLP"]
    adj_edge2 = data["PAP"]
    adj_fusion1 = adj_edge1 + adj_edge2
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 2] = 0
    adj_fusion[adj_fusion == 2] = 1

    adj1 = data["PLP"] + np.eye(data["PLP"].shape[0]) * sc
    adj2 = data["PAP"] + np.eye(data["PAP"].shape[0]) * sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_dblp(sc=3):
    data = pkl.load(open("utils/data/dblp.pkl", "rb"))
    label = data['label']

    adj_edge1 = data["PAP"]
    adj_edge2 = data["PPrefP"]
    adj_edge3 = data["PATAP"]
    adj_fusion1 = adj_edge1 + adj_edge2 + adj_edge3
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 3] = 0
    adj_fusion[adj_fusion == 3] = 1

    adj1 = data["PAP"] + np.eye(data["PAP"].shape[0]) * sc
    adj2 = data["PPrefP"] + np.eye(data["PPrefP"].shape[0]) * sc
    adj3 = data["PATAP"] + np.eye(data["PATAP"].shape[0]) * sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2, adj3]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_imdb(sc=3):
    data = pkl.load(open("utils/data/imdb.pkl", "rb"))
    label = data['label']
    ###########################################################
    adj_edge1 = data["MDM"]
    adj_edge2 = data["MAM"]
    adj_fusion1 = adj_edge1 + adj_edge2
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 2] = 0
    adj_fusion[adj_fusion == 2] = 1
    ############################################################
    adj1 = data["MDM"] + np.eye(data["MDM"].shape[0]) * sc
    adj2 = data["MAM"] + np.eye(data["MAM"].shape[0]) * sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_amazon(sc=3):
    data = pkl.load(open("utils/data/amazon.pkl", "rb"))
    label = data['label']

    adj_edge1 = data["IVI"]
    adj_edge2 = data["IBI"]
    adj_edge3 = data["IOI"]
    adj_fusion1 = adj_edge1 + adj_edge2 + adj_edge3
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 3] = 0
    adj_fusion[adj_fusion == 3] = 1

    adj1 = data["IVI"] + np.eye(data["IVI"].shape[0]) * sc
    adj2 = data["IBI"] + np.eye(data["IBI"].shape[0]) * sc
    adj3 = data["IOI"] + np.eye(data["IOI"].shape[0]) * sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2, adj3]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_freebase(sc=None):
    type_num = 3492
    ratio = [20, 40, 60]
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "utils/data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_m = sp.eye(type_num)

    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    # pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.FloatTensor(label)
    adj_list = [mam, mdm, mwm]

    # pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]
    return adj_list, feat_m, label, train[0], val[0], test[0]


def load_pubmed(train_ratio=1/3,val_ratio=1/3):
    data = sio.loadmat('./utils/data/PubMed.mat')

    label = data['labels'][0]
    features = data['features']
    adj_list = list(data['adjs_sparse'][0])
    labels_mask = data['labels_mask'][0]

    label_num = np.sum(labels_mask)
    train_num = int(label_num*train_ratio)
    val_num = int(label_num*val_ratio)

    index = np.where(labels_mask == 1)[0]
    np.random.shuffle(index)

    idx_train = index[:train_num]
    idx_val = index[train_num:train_num+val_num]
    idx_test = index[train_num+val_num:]

    # adj_fusion1 = (adj_list[0]+adj_list[1]+adj_list[2]+adj_list[3])
    # adj_fusion = np.array(adj_fusion1.todense().copy())
    # adj_fusion[adj_fusion < 4] = 0
    # adj_fusion[adj_fusion == 4] = 1

    return adj_list, features, label, idx_train, idx_val, idx_test#, adj_fusion



class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_abide(label_rate = 0.8):
    data = np.load('utils/data/ABIDE/abide.npy', allow_pickle=True).item()
    final_fc = data["timeseires"]
    final_pearson = data["corr"]
    labels = data["label"]
    site = data['site']
    _, _, timeseries = final_fc.shape
    _, node_size, node_feature_size = final_pearson.shape
    scaler = StandardScaler(mean=np.mean(final_fc), std=np.std(final_fc))
    final_fc = scaler.transform(final_fc)
    final_fc, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_fc, final_pearson, labels)]
    # split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    kf = StratifiedShuffleSplit(n_splits=5, train_size=label_rate, random_state=1)
    train_list = []
    val_list = []
    test_list = []
    for train_index, test_index in kf.split(final_fc, site):
        train_list.append(list(train_index))
        test_list.append(list(test_index)[:int(len(list(test_index))/2)])
        val_list.append(list(test_index)[int(len(list(test_index))/2):])
    return final_fc, final_pearson, labels, train_list, test_list, val_list


def compute_ppr(adj: Tensor, alpha=0.2, self_loop=True) -> Tensor:  ### PPR (personalized PageRank)：
    adj = adj.numpy()
    if self_loop:
        adj = adj + np.eye(adj.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(adj, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, adj), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return torch.tensor(alpha * inv((np.eye(adj.shape[0]) - (1 - alpha) * at)),
                        dtype=torch.float32)  # a(I_n-(1-a)A~)^-1


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    try:
        return features.todense()
    except:
        return features


def row_normalize(A):
    """Row-normalize dense matrix"""
    eps = 2.2204e-16
    rowsum = A.sum(dim=-1).clamp(min=0.) + eps
    r_inv = rowsum.pow(-1)
    A = r_inv.unsqueeze(-1) * A
    return A


def row_normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch2dgl(graph):
    N = graph.shape[0]
    if graph.is_sparse:
        graph_sp = graph.coalesce()
    else:
        graph_sp = graph.to_sparse()
    edges_src = graph_sp.indices()[0]
    edges_dst = graph_sp.indices()[1]
    edges_features = graph_sp.values()
    graph_dgl = dgl.graph((edges_src, edges_dst), num_nodes=N)
    # graph_dgl.edate['w'] = edges_features
    return graph_dgl


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def mask_edge(graph, mask_prob):
    E = graph.number_of_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def RA(graph, x, feat_drop_rate, edge_mask_rate):
    n_node = graph.number_of_nodes()

    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = drop_feature(x, feat_drop_rate)

    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]
    ng.add_edges(nsrc, ndst)

    return ng, feat


def separate_adj(org_adj, idx_train, idx_test, idx_val):
    train_adj = torch.clone(org_adj)
    test_adj = torch.clone(org_adj)
    val_adj = torch.clone(org_adj)

    train_adj[:, idx_test] = 0
    train_adj[:, idx_val] = 0
    train_adj[idx_test, :] = 0
    train_adj[idx_val, :] = 0

    test_adj[:, idx_train] = 0
    test_adj[:, idx_val] = 0
    test_adj[idx_train, :] = 0
    test_adj[idx_val, :] = 0

    val_adj[:, idx_train] = 0
    val_adj[:, idx_test] = 0
    val_adj[idx_train, :] = 0
    val_adj[idx_test, :] = 0

    I = torch.eye(org_adj.shape[1]).to(org_adj.device)

    train_adj_normal = row_normalize(train_adj + I)
    test_adj_normal = row_normalize(test_adj + I)
    val_adj_normal = row_normalize(val_adj + I)

    return train_adj_normal, test_adj_normal, val_adj_normal


def n_fold_split(n, label, feature, train_rate):
    kf = StratifiedShuffleSplit(n_splits=n, train_size=train_rate)
    train_list = []
    test_list = []
    for train_index, test_index in kf.split(feature.cpu().numpy(), label.cpu().numpy()):
        train_list.append(list(train_index))
        test_list.append(list(test_index))

    return train_list, test_list, test_list
