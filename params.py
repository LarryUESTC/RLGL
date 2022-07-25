import argparse


def parse_args(dataset):
    parser = argparse.ArgumentParser()
    if dataset == 'acm':
        parser.add_argument('--cfg', type=int, default=[512, 128], help='hidden dimension')
        parser.add_argument('--dataset', nargs='?', default='acm')
        parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
        parser.add_argument('--sparse', type=bool, default=True, help='sparse adjacency matrix')
        parser.add_argument('--nb_epochs', type=int, default=400, help='the number of epochs')
        parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        parser.add_argument('--patience', type=int, default=50, help='patience for early stopping')
        parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')
        parser.add_argument('--seed', type=int, default=0, help='the seed to use')
        parser.add_argument('--test_epo', type=int, default=50, help='test_epo')
        parser.add_argument('--test_lr', type=int, default=0.01, help='test_lr')
        parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
        parser.add_argument('--num_clusters', type=float, default=3, help='')
        parser.add_argument('--neg_num', type=int, default=2, help='the number of negtives')
        parser.add_argument('--margin1', type=float, default=0.8, help='')
        parser.add_argument('--margin2', type=float, default=0.4, help='')
        parser.add_argument('--w_s', type=float, default=10, help='weight of loss L_s')
        parser.add_argument('--w_c', type=float, default=10, help='weight of loss L_c')
        parser.add_argument('--w_ms', type=float, default=1, help='weight of loss L_ms')
        parser.add_argument('--w_u', type=float, default=1, help='weight of loss L_u')
    elif dataset == 'dblp':
        parser.add_argument('--cfg', type=int, default=[512, 128], help='hidden dimension')
        parser.add_argument('--dataset', nargs='?', default='dblp')
        parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
        parser.add_argument('--sparse', type=bool, default=True, help='sparse adjacency matrix')
        parser.add_argument('--nb_epochs', type=int, default=7000, help='the number of epochs')
        parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
        parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')
        parser.add_argument('--seed', type=int, default=0, help='the seed to use')
        parser.add_argument('--test_epo', type=int, default=100, help='test_epo')
        parser.add_argument('--test_lr', type=int, default=0.2, help='test_lr')
        parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
        parser.add_argument('--neg_num', type=int, default=5, help='the number of negtives')
        parser.add_argument('--margin1', type=float, default=0.9, help='')
        parser.add_argument('--margin2', type=float, default=0.2, help='')
        parser.add_argument('--w_s', type=float, default=10, help='weight of loss L_s')
        parser.add_argument('--w_c', type=float, default=10, help='weight of loss L_c')
        parser.add_argument('--w_ms', type=float, default=5, help='weight of loss L_ms')
        parser.add_argument('--w_u', type=float, default=6, help='weight of loss L_u')

    elif dataset == 'imdb':
        parser.add_argument('--cfg', type=int, default=[512, 256], help='hidden dimension')
        parser.add_argument('--dataset', nargs='?', default='imdb')
        parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
        parser.add_argument('--sparse', type=bool, default=True, help='sparse adjacency matrix')
        parser.add_argument('--nb_epochs', type=int, default=400, help='the number of epochs')
        parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')  # imdb 0.0005
        parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
        parser.add_argument('--gpu_num', type=int, default=1, help='the id of gpu to use')
        parser.add_argument('--seed', type=int, default=0, help='the seed to use')
        parser.add_argument('--test_epo', type=int, default=50, help='test_epo')
        parser.add_argument('--test_lr', type=int, default=0.3, help='test_lr')
        parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
        parser.add_argument('--neg_num', type=int, default=6, help='the number of negtives')
        parser.add_argument("--learning_rate", default=1e-4, type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--lr_scheduler", type=str, default="cosine",
                            choices=["linear", "cosine", "cosine_restart"],
                            help="Choose the optimizer to use. Default RecAdam.")
        parser.add_argument('--margin1', type=float, default=0.5, help='')
        parser.add_argument('--margin2', type=float, default=0.3, help='')
        parser.add_argument('--w_s', type=float, default=10, help='weight of loss L_s')
        parser.add_argument('--w_c', type=float, default=15, help='weight of loss L_c')
        parser.add_argument('--w_ms', type=float, default=10, help='weight of loss L_ms')
        parser.add_argument('--w_u', type=float, default=0.01, help='weight of loss L_u')

    elif dataset == 'amazon':
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=int, default=[512, 256], help='hidden dimension')
        parser.add_argument('--dataset', nargs='?', default='amazon')
        parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
        parser.add_argument('--sparse', type=bool, default=True, help='sparse adjacency matrix')
        parser.add_argument('--nb_epochs', type=int, default=6000, help='the number of epochs')
        parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
        parser.add_argument('--gpu_num', type=int, default=3, help='the id of gpu to use')
        parser.add_argument('--seed', type=int, default=0, help='the seed to use')
        parser.add_argument('--test_epo', type=int, default=100, help='test_epo')
        parser.add_argument('--test_lr', type=int, default=0.1, help='test_lr')
        parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
        parser.add_argument('--neg_num', type=int, default=3, help='the number of negtives')
        parser.add_argument('--margin1', type=float, default=0.6, help='')
        parser.add_argument('--margin2', type=float, default=0.2, help='')
        parser.add_argument('--w_s', type=float, default=1, help='weight of loss L_s')
        parser.add_argument('--w_c', type=float, default=20, help='weight of loss L_c')
        parser.add_argument('--w_ms', type=float, default=0.1, help='weight of loss L_ms')
        parser.add_argument('--w_u', type=float, default=20, help='weight of loss L_u')

    else:
        parser.add_argument('--cfg', type=int, default=[512, 128], help='hidden dimension')
        parser.add_argument('--dataset', nargs='?', default=dataset)
        parser.add_argument('--sparse', type=bool, default=True, help='sparse adjacency matrix')
        parser.add_argument('--nb_epochs', type=int, default=400, help='the number of epochs')
        parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        parser.add_argument('--patience', type=int, default=40, help='patience for early stopping')
        parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')
        parser.add_argument('--seed', type=int, default=0, help='the seed to use')
        parser.add_argument('--test_epo', type=int, default=50, help='test_epo')
        parser.add_argument('--test_lr', type=int, default=0.01, help='test_lr')
        parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
        parser.add_argument('--num_clusters', type=float, default=7, help='')
        parser.add_argument('--random_aug_feature', type=float, default= 0.2, help='RA feature')
        parser.add_argument('--random_aug_edge', type=float, default=0.2, help='RA graph')
        parser.add_argument('--alpha', type=float, default= 1, help='loss alpha')
        parser.add_argument('--beta', type=float, default=0.1, help='loss beta')


    return parser.parse_known_args()


def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)
