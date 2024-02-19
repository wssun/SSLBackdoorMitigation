import copy
import torch
import torchvision
from torch import nn
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
import torch.nn.functional as F

class MoCoBase(nn.Module):
    def __init__(self, arch='resnet18'):
        super(MoCoBase, self).__init__()
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


class MoCo(nn.Module):
    def __init__(self, arch='resnet18'):
        super().__init__()
        self.f = MoCoBase(arch)
        self.projection_head = MoCoProjectionHead(512, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.f)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query = self.f(x)
        
        # query = self.projection_head(query)
        return F.normalize(query, dim=-1)
     
    def forward_momentum(self, x):
        key = self.backbone_momentum(x)
        key = self.projection_head_momentum(key).detach()
        return key