import copy

import torch
import torchvision
from torch import nn
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum


class BYOLBase(nn.Module):
    def __init__(self, arch='resnet18'):
        super(BYOLBase, self).__init__()
        self.f = []
        if arch == 'resnet18':
            model_name = torchvision.models.resnet18()
        else:
            raise ValueError('Not supported!')
        for name, module in model_name.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        return feature


class BYOL(nn.Module):
    def __init__(self, arch='resnet18'):
        super().__init__()
        self.f = BYOLBase(arch)
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.f)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.f(x)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z
    
    def forward_p(self, x):
        y = self.f(x)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return y
    
