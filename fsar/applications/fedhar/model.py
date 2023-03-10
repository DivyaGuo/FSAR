import torch.nn as nn
from torch.nn import init
from torchvision import models

from easyfl.models import BaseModel
from easyfl.models.stgcn import Model as stgcnBackbone

import torch


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def get_classifier(class_num, num_bottleneck=512):
    classifier = []
    classifier += [nn.Linear(num_bottleneck, class_num)]
    classifier = nn.Sequential(*classifier)
    classifier.apply(weights_init_classifier)
    return classifier

def get_unshareC():
    A = torch.load('/root/FL-HAR/A.pt')
    C = nn.Parameter(A)
    nn.init.constant_(C, 1e-6)
    return C


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        if class_num > 0:
            classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x


class CBlock(nn.Module):
    def __init__(self, class_num=0):
        super(CBlock, self).__init__()
        A = torch.load('/root/FL-HAR/A.pt')
        C = nn.Parameter(A)
        nn.init.constant_(C, 1e-6)
        self.unshareC = C

    def forward(self, x):
        x = self.unshareC(x)
        return x


# ST-GCN Model
class Model(BaseModel):

    def __init__(self, class_num=0, droprate=0.5, stride=2):
        super(Model, self).__init__()

        graph_args = {
            "layout": "ntu-rgb+d",
            "strategy": "spatial",
        }
        model_ft = stgcnBackbone(in_channels=3, hidden_channels=16,
                                hidden_dim=256, num_class=class_num,
                                graph_args=graph_args,
                                edge_importance_weighting=True)
        self.model = model_ft

        self.class_num = class_num
        self.classifier = ClassBlock(256, class_num, droprate)
        self.model.fc = self.classifier

        # self.unshareC = get_unshareC()
        # self.model.unshareC = self.unshareC

    def forward(self, x, mode="", start_level=-1, NM_for_train_model=None):
        x = self.model(x, mode=mode, start_level=start_level, NM_for_train_model=NM_for_train_model)
        return x


