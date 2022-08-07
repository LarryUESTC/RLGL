import time
from utils import process
from models.embedder import embedder_image
from models.Layers import act_layer
from timm.models.layers import DropPath
from .gcn_lib import Grapher
import os
from evaluate import evaluate, accuracy
from models.Layers import SUGRL_Fast, GNN_Model
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features  # if there is not out_feature, use in_feature as output
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)


class Downsample(nn.Module):
    """ Convolution-based downsample
    """

    def __init__(self, in_dim=3, out_dim=768):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super(Stem, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),  # now, img_size = img_size/2
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),  # now, img_size = img_size/4
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class ViG_Model(nn.Module):
    def __init__(self, num_classes=10, cfg=None, k=5, blocks=None, img_size=None, conv='mr', norm='batch', act='gelu',
                 bias=True, dropout=0.0, use_dialation=True, epsilon=0.2, use_stochastic=False, drop_path_rate=0.0):
        super(ViG_Model, self).__init__()
        # to be config
        self.n_blocks = sum(blocks)
        reduce_ratios = torch.ones(self.n_blocks, dtype=torch.int).tolist()
        dpr = torch.linspace(0, drop_path_rate, self.n_blocks).tolist()  # stochastic depth decay rule
        num_knn = torch.linspace(k, k, self.n_blocks, dtype=torch.int).tolist()  # number of knn's k
        max_dilation = 49 // max(num_knn)

        self.stem = Stem(out_dim=cfg[0], act=act)  # size become 1/4
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg[0], img_size // 4, img_size // 4))
        HW = (img_size // 4) * (img_size // 4)

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(cfg[i - 1], cfg[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    nn.Sequential(
                        Grapher(cfg[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm, bias,
                                use_stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx], relative_pos=True),
                        FFN(cfg[i], cfg[i] * 4, act=act, drop_path=dpr[idx])
                    )
                ]
                idx += 1
        self.backbone = nn.Sequential(*self.backbone)
        self.prediction = nn.Sequential(
            nn.Conv2d(cfg[-1], 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            act_layer(act),
            nn.Dropout(dropout),
            nn.Conv2d(1024, num_classes, 1, bias=True)
        )
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)


class PyramidViG(embedder_image):
    def __init__(self, args):
        super(PyramidViG, self).__init__(args)
        self.args = args
        self.model = ViG_Model(num_classes=args.num_classes, cfg=args.cfg, blocks=args.blocks, k=5,
                               img_size=args.image_size).to(args.device)

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.args.lr * (0.1 ** (epoch // 100))
        for para_group in optimizer.param_groups:
            para_group['lr'] = lr

    def training(self):
        print("Started training...")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        criterion = nn.CrossEntropyLoss().to(self.args.device)

        cnt_wait = 0
        best = 1e-9
        output_acc = 1e-9
        stop_epoch = 0

        start = time.time()

        for epoch in range(self.args.nb_epochs):
            self.model.train()
            loss_num = torch.zeros(1).to(self.args.device)
            accu_num = torch.zeros(1).to(self.args.device)
            sample_num = 0

            for iteration, (inputs, labels) in enumerate(self.train_dl):
                if inputs.size(3) == 3:
                    inputs = inputs.permute(0, 3, 1, 2)
                inputs = inputs.type(torch.FloatTensor)
                targets = labels.type(torch.LongTensor)
                sample_num += inputs.shape[0]

                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)
                inputs = Variable(inputs)
                targets = Variable(targets)

                optimizer.zero_grad()

                y = self.model(inputs)
                pred_classes = torch.max(y, dim=1)[1]
                accu_num += torch.eq(pred_classes, targets).sum()

                loss = criterion(y, targets)
                loss_num += loss.detach()
                loss.backward()
                optimizer.step()

                if iteration % 100 == 0 and iteration != 0:
                    print("Epoch[{}]({}/{}):  Loss_H: {:.4f}".format(epoch, iteration, len(self.train_dl), loss.item()))
            self.adjust_learning_rate(optimizer, epoch)
            print("the learning rate of %dth: %f" % (epoch, optimizer.param_groups[0]['lr']))
            ################STA|Eval|###############
            if epoch+1 % 5 == 0 and epoch != 0:
                self.model.eval()
                train_Acc = 100. * np.float64(accu_num) / sample_num
                train_Loss = loss_num.item() / (iteration + 1)
                print(f"train_Acc：{train_Acc} train_Loss{train_Loss}")
                test_Acc, test_Loss = self.evaluate()
                print(f"test_Acc：{test_Acc} test_Loss{test_Loss}")

                stop_epoch = epoch
                if test_Acc > best:
                    best = test_Acc
                    output_acc = test_Acc
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

    def evaluate(self):
        # V1-shift model to eval model, then define 2 varis to reserve val+ACCs&Losses
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.model.eval()
        loss_num = torch.zeros(1).to(self.args.device)
        accu_num = torch.zeros(1).to(self.args.device)
        sample_num = 0
        # V2-evaluate the model
        for iteration, (inputs, labels) in enumerate(self.test_dl):
            # I0-load data
            if inputs.size(3) == 3:
                inputs = inputs.permute(0, 3, 1, 2)
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.LongTensor)
            sample_num += inputs.shape[0]
            labels = labels.to(self.args.device)

            # I2-model evaluating
            with torch.no_grad():
                inputs = inputs.to(self.args.device)
                inputs = Variable(inputs)
                labels = Variable(labels)
                y = self.model(inputs)
            pred_classes = torch.max(y, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum()

            # I3-calculate loss
            loss = criterion(y, labels)
            loss_num += loss.item()

        # V5 Integrate training Accs&Losses
        val_Acc = 100. * np.float64(accu_num) / sample_num
        val_Loss = loss_num.item() / (iteration + 1)
        return val_Acc, val_Loss
