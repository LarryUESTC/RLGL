import numpy as np
import random as random
import torch
from params import parse_args
import models
from utils.sql_writer import WriteToDatabase, get_primary_key_and_value, get_columns, merge_args_and_dict, \
    merge_args_and_config
from statistics import mean
import socket, os
import gc
import copy


def main_one(config, checkpoint_dir=None):
    ################STA|SQL|###############
    current_args = copy.deepcopy(args)
    current_args = merge_args_and_config(current_args, config)
    host_name = socket.gethostname()
    db_input_dir = {"name": ["text", host_name + os.path.split(__file__)[-1][:-3]],
                    "epoch": ["integer", None],
                    "stop_epoch": ["integer", None],
                    "seed": ["integer", None], }  # Larry: set key
    PRIMARY_KEY, PRIMARY_VALUE = get_primary_key_and_value(
        merge_args_and_dict(copy.deepcopy(db_input_dir), vars(current_args)))
    REFRESH = False
    OVERWRITE = True

    test_metrics = {
        "acc": None,
        "time": None,
    }
    train_metrics = {
        "acc": None,
        "time": None,
    }
    val_metrics = {
        "acc": None,
        "time": None,
    }

    TABLE_NAME = 'main_RLGL_' + current_args.task + '_' + current_args.method + '_0'
    # try:
    #     writer = WriteToDatabase({'host': "postgres.kongfei.life", "port": "",
    #                               "database": "pengliang", "user": "pengliang", "password": ""},
    #                              TABLE_NAME,
    #                              PRIMARY_KEY,
    #                              get_columns(train_metrics, val_metrics, test_metrics),
    #                              PRIMARY_VALUE,
    #                              PRIMARY_VALUE,
    #                              REFRESH,
    #                              OVERWRITE)
    #     writer.init()
    # except:
    #     print("Keys not matched in current table, pls check KEY, or network error")
    #     print("Change TABLE_NAME to create a new table")
    ################END|SQL|###############

    current_args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ACC_seed = []
    Time_seed = []
    for seed in range(0, 5):

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        current_args.seed = seed
        method_fun = models.getmodel(current_args.method)
        embedder = method_fun(copy.deepcopy(current_args))
        test_acc, training_time, stop_epoch = embedder.training()

        ################STA|write one|###############
        # try:
        #     writer_matric_seed = {'epoch': -1, "seed": seed, "test_time": training_time, "stop_epoch": stop_epoch, }
        #     writer.write(writer_matric_seed, {"test_acc": test_acc, })
        # except:
        #     print("DataBase is not available, it's fine")
        ################END|write one|###############
        ACC_seed.append(test_acc)
        # St_seed.append(np.mean(test_st))
        Time_seed.append(training_time)
        torch.cuda.empty_cache()
        gc.collect()

    ################STA|write seed|###############
    # try:
    #     writer_matric_seed = {'epoch': -2, "seed": -2, "test_time": mean(Time_seed), "stop_epoch": -2}
    #     writer.write(writer_matric_seed, {"test_acc": mean(ACC_seed), })
    # except:
    #     print("DataBase is not available, it's fine")
    ################END|write seed|###############


def main(args):
    # param set
    ################STA|set tune param|###############
    config = {
        # 'nb_epochs': 1000,
        # 'cfg': [128, 128],
        # 'random_aug_feature': 0.2,
        # 'batch': 128
        # 'beta': 10,
        # 'gama': 1,
        # 'knn': 5,
        # 'delta': 0.2
    }
    main_one(config)
    ################END|set tune param|###############

#todo list
# 1. method name
# 2. liner
# 3. local perseve loss
# 4.
if __name__ == '__main__':
    task = 'Unsup'  # choice:Semi Unsup Sup Rein Noise ImgCls Brain
    method = 'CCAMGRL'  # choice: Gcn ViG GDP GcnMixup SelfCons GcnCR offlineRLG SelfBrain SelfBrainMLP CCAMGRL
    dataset = 'Acm'  # choice:Cora CiteSeer PubMed CIFAR10 abide Acm
    args = parse_args(task, method, dataset)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args)
