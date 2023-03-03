import numpy as np
import random as random
import torch
from params import parse_args
import models
from ray import tune
import ray
from ray.tune.integration.torch import DistributedTrainableCreator
from ray.tune.suggest.hebo import HEBOSearch
from utils.sql_writer import WriteToDatabase, get_primary_key_and_value, get_columns, merge_args_and_dict, \
    merge_args_and_config
from statistics import mean
import socket, os
import gc
import copy

TUNE = True


def main_one(config, checkpoint_dir=None):
    ################STA|SQL|###############
    current_args = copy.deepcopy(args)
    current_args = merge_args_and_config(current_args, config)
    host_name = socket.gethostname()
    db_input_dir = {"name": ["text", host_name + os.path.split(__file__)[-1][:-3]],
                    "epoch": ["integer", None],
                    #"stop_epoch": ["integer", None],
                    "seed": ["integer", None], }  # Larry: set key
    PRIMARY_KEY, PRIMARY_VALUE = get_primary_key_and_value(
        merge_args_and_dict(copy.deepcopy(db_input_dir), vars(current_args)))
    REFRESH = False
    OVERWRITE = True

    test_metrics = {
        "acc": None,
        "acc_std": None,
        "macro_f1": None,
        "macro_f1_std": None,
        "micro_f1": None,
        "micro_f1_std": None,
    }
    train_metrics = {
        "acc": None,
        "time": None,
    }
    val_metrics = {
        "acc": None,
        "time": None,
    }

    TABLE_NAME = 'CCAMGRL_hyperParameter'
    try:
        writer = WriteToDatabase({'host': "121.48.161.92", "port": "40201",
                                  "database": "wangxin", "user": "wangxin", "password": "262300aa"},
                                 TABLE_NAME,
                                 PRIMARY_KEY,
                                 #get_columns(train_metrics, val_metrics, test_metrics),
                                 get_columns({'time'}, {}, test_metrics),
                                 PRIMARY_VALUE,
                                 PRIMARY_VALUE,
                                 REFRESH,
                                 OVERWRITE)
        writer.init()
    except:
        print("Keys not matched in current table, pls check KEY, or network error")
        print("Change TABLE_NAME to create a new table")
    ################END|SQL|###############

    # if args.gpu_num == -1:
    #     args.device = 'cpu'
    # else:
    #     args.device = torch.device("cuda:" + str(args.gpu_num) if torch.cuda.is_available() else "cpu")

    current_args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ACC_seed = []
    Time_seed = []
    for seed in range(2021, 2024):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        method_fun = models.getmodel(current_args.method)
        embedder = method_fun(copy.deepcopy(current_args))
        # from models.NodeClas.Semi_SELFCONS_plus import SELFCONS_plus
        # embedder = SELFCONS_plus(copy.deepcopy(current_args))

        acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, k1, k2, st,training_time = embedder.training()

        ################STA|write one|###############
        writer_matric_seed = {'epoch': args.nb_epochs, "seed": seed, "train_time": training_time}
        writer.write(writer_matric_seed,
                     {
                         "test_acc": acc,
                         "test_acc_std": acc_std,
                         "test_macro_f1": macro_f1,
                         "test_macro_f1_std": macro_f1_std,
                         "test_micro_f1": micro_f1,
                         "test_micro_f1_std": micro_f1_std,
                     }
                     )
        ################END|write one|###############
        ACC_seed.append(acc)
        #St_seed.append(np.mean(test_st))
        Time_seed.append(training_time)
        torch.cuda.empty_cache()
        gc.collect()

    ################STA|write seed|###############
    # writer_matric_seed = {'epoch': -2, "seed": -2, "test_time": mean(Time_seed), "stop_epoch": -2
    #                       }
    # writer.write(writer_matric_seed,
    #              {
    #                  "test_acc": mean(ACC_seed),
    #              }
    #              )
    ################END|write seed|###############
    if TUNE:
        tune.report(test_sum=mean(ACC_seed))


def main(args):
    # param set
    ################STA|set tune param|###############
    if TUNE:
        os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,4,5,0,1"
        ray.init(num_gpus=2)
        config = {
            'nb_epochs': tune.choice([500,1000,1500]),
            # 'patience': tune.choice([100, 200, 300]),
            'lr': tune.grid_search([0.0005, 0.0001, 0.005,0.001]),
            'w_c': tune.grid_search([0.001, 0.01, 0.1,1,10]),
            'w_l': tune.grid_search([0.01, 0.1,1,5,10]),
            'tau': tune.grid_search([0.5,1.0,5]),
            'A_r': tune.grid_search([2,3,4]),
            'dropout': tune.grid_search([0.2,0.5,0.7]),
            'sc': 0,
            'test_epo': 100,
            'test_lr': 0.01,
            'cfg': [512, 512, 256, 256, 128, 128],
        }
        # search_alg = HEBOSearch(metric='test_sum', mode='max')
        distributed_ray_run = DistributedTrainableCreator(
            main_one,
            backend='nccl',
            num_gpus_per_worker=0.125,
            num_workers=1,
        )
        # tune.run(distributed_ray_run, config=config, num_samples=5000 )
        tune.run(distributed_ray_run, config=config, verbose=1)
        # search_alg.save('checkpoint_alg')
    else:
        config = {
            'nb_epochs': 100,
            'lr': 0.01,
            'w_c': 1,
            'w_l': 1,
            'tau': 1,
            'A_r': 2,
            'dropout': 0.5,
            'sc': 0,
            'test_epo': 100,
            'test_lr': 0.01,
            'cfg': [256, 256, 128, 128],
        }
        main_one(config)
    ################END|set tune param|###############


if __name__ == '__main__':
    task = 'Unsup'  # choice:Semi Unsup Sup Rein Noise ImgCls Brain
    method = 'CCAMGRL'  # choice: Gcn ViG GDP GcnMixup SelfCons GcnCR offlineRLG SelfBrain SelfBrainMLP CCAMGRL
    dataset = 'Imdb'  # choice:Cora CiteSeer PubMed CIFAR10 abide Acm Imdb Dblp Freebase Amazon
    args = parse_args(task, method, dataset)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args)
