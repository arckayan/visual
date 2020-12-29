import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn.utils.rnn import pack_padded_sequence


class Resnet(nn.Module):
    """
    Resnet is the underlying image extraction model for vqa
    """
    def __init__(self):
        super(Resnet, self).__init__()
        self.model = models.resnet152(pretrained=True)

        def hook(module, x, y):
            self.extracted = y

        self.model.layer4.register_forward_hook(hook)

    def forward(self, x):
        self.model(x)
        return self.extracted


class ShowAskAttend(nn.Module):
    """
    Paper - https://arxiv.org/abs/1704.03162
    """
    def __init__(self, config):
        super(ShowAskAttend, self).__init__()
        self.image = Resnet()

    def forward(self, x):
        v, q, a = x
        return  self.image(v)
