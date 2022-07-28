import argparse

################STA|Semi-supervised Task|###############

class Semi(object):
    def __init__(self, method, dataset):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataset', nargs='?', default=dataset)
        self.parser.add_argument('--method', nargs='?', default=method)
        self.parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--patience', type=int, default=40, help='patience for early stopping')
        self.parser.add_argument('--seed', type=int, default=0, help='the seed to use')
        self.parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
        self.parser.add_argument('--random_aug_feature', type=float, default=0.2, help='RA feature')
        self.parser.add_argument('--random_aug_edge', type=float, default=0.2, help='RA graph')
        self.args, _ = self.parser.parse_known_args()

    def replace(self):
        pass

    def get_parse(self):
        return self.args

class Semi_Gcn(Semi):
    def __init__(self, method, dataset):
        super(Semi_Gcn, self).__init__(method, dataset)
        ################STA|add new params here|###############
        # self.parser.add_argument('--random_aug_edge', type=float, default=0.2, help='RA graph')
        ################END|add new params here|###############
        self.args, _ = self.parser.parse_known_args()

        ################STA|replace params here|###############
        self.replace()
        ################END|replace params here|###############

    def replace(self):
        super(Semi_Gcn, self).replace()
        self.args.__setattr__('method', 'Gcn')
        self.args.__setattr__('lr', 0.05)

class Semi_Gcn_Cora(Semi_Gcn):
    def __init__(self, method, dataset):
        super(Semi_Gcn, self).__init__(method, dataset)
        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Semi_Gcn_Cora, self).replace()
        self.args.__setattr__('dataset', 'Cora')
        self.args.__setattr__('lr', 0.01)

################END|Semi-supervised Task|###############




################STA|unsupervised Task |###############

class Unsup(object):
    def __init__(self, method, dataset):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataset', nargs='?', default=dataset)
        self.parser.add_argument('--method', nargs='?', default=method)
        self.parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--patience', type=int, default=40, help='patience for early stopping')
        self.parser.add_argument('--seed', type=int, default=0, help='the seed to use')
        self.parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
        self.parser.add_argument('--random_aug_feature', type=float, default=0.2, help='RA feature')
        self.parser.add_argument('--random_aug_edge', type=float, default=0.2, help='RA graph')

        self.parser.add_argument('--cfg', type=int, default=[512, 128], help='hidden dimension')
        self.parser.add_argument('--nb_epochs', type=int, default=400, help='the number of epochs')
        self.parser.add_argument('--test_epo', type=int, default=50, help='test_epo')
        self.parser.add_argument('--test_lr', type=int, default=0.01, help='test_lr')

        self.args, _ = self.parser.parse_known_args()

    def replace(self):
        pass

    def get_parse(self):
        return self.args

class Unsup_E2sgrl(Unsup):
    def __init__(self,  method, dataset):
        super(Unsup_E2sgrl,self).__init__(method, dataset)

        self.parser.add_argument('--neg_num', type=int, default=2, help='the number of negtives')
        self.parser.add_argument('--margin1', type=float, default=0.8, help='')
        self.parser.add_argument('--margin2', type=float, default=0.4, help='')
        self.parser.add_argument('--w_s', type=float, default=10, help='weight of loss L_s')
        self.parser.add_argument('--w_c', type=float, default=10, help='weight of loss L_c')
        self.parser.add_argument('--w_ms', type=float, default=1, help='weight of loss L_ms')
        self.parser.add_argument('--w_u', type=float, default=1, help='weight of loss L_u')

        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_E2sgrl, self).replace()
        # self.args.__setattr__('dataset', 'acm')

class Unsup_E2sgrl_Acm(Unsup_E2sgrl):
    def __init__(self, method, dataset):
        super(Unsup_E2sgrl_Acm,self).__init__(method, dataset)

        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_E2sgrl_Acm, self).replace()
        self.args.__setattr__('dataset', 'acm')

class Unsup_E2sgrl_Dblp(Unsup_E2sgrl):
    def __init__(self, method, dataset):
        super(Unsup_E2sgrl_Dblp,self).__init__(method, dataset)

        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_E2sgrl_Dblp, self).replace()
        self.args.__setattr__('dataset', 'dblp')

class Unsup_E2sgrl_Imdb(Unsup_E2sgrl):
    def __init__(self, method, dataset):
        super(Unsup_E2sgrl_Imdb,self).__init__(method, dataset)

        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_E2sgrl_Imdb, self).replace()
        self.args.__setattr__('dataset', 'imdb')

class Unsup_E2sgrl_Amazon(Unsup_E2sgrl):
    def __init__(self, method, dataset):
        super(Unsup_E2sgrl_Amazon,self).__init__(method, dataset)

        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_E2sgrl_Amazon, self).replace()
        self.args.__setattr__('dataset', 'amazon')

################END|unsupervised Task |###############




params_key = {
'Semi': Semi,
'Semi_Gcn': Semi_Gcn,
'Semi_Gcn_Cora': Semi_Gcn_Cora,
'Unsup': Unsup,
'Unsup_E2sgrl': Unsup_E2sgrl,
'Unsup_E2sgrl_Acm': Unsup_E2sgrl_Acm,
'Unsup_E2sgrl_Dblp': Unsup_E2sgrl_Acm,
'Unsup_E2sgrl_Imdb': Unsup_E2sgrl_Acm,
'Unsup_E2sgrl_Amazon': Unsup_E2sgrl_Acm,
}

def parse_args(task, method, dataset):

    name_3 = task + '_' + method + '_' + dataset
    name_2 = task + '_' + method
    name_1 = task

    if name_3 in params_key:
        return params_key[name_3](method, dataset).get_parse()
    elif name_2 in params_key:
        return params_key[name_2](method, dataset).get_parse()
    elif name_1 in params_key:
        return params_key[name_1](method, dataset).get_parse()
    else:
        return None

def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)
