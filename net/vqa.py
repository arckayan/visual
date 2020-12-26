import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.model = models.resnet152(pretrained=True)

        def hook(module, x, y):
            self.extracted = y

        self.model.layer4.register_forward_hook(hook)

    def forward(self, x):
        self.model(x)
        return self.extracted

class Vqa(nn.Module):
    def __init__(self):
        super(Vqa, self).__init__()
        self.image = Resnet()

    def forward(self, x):
        v, q, a = x
        return  self.image(v)
