#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 2 19:29:18 2017

@author: Go-hiroaki

network models in pytorch
"""
import sys
sys.path.append('./models')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, in_ch=1, nb_layers=3, base_kernel=64):
        super(MLP, self).__init__()
        # set up kernels
        kernels = [in_ch]
        kernels += [base_kernel * (2**i) for i in range(0, nb_layers)]
        # add layers
        layers = []
        for i in range(nb_layers):
            layers.extend([
                nn.Conv2d(kernels[i], kernels[i + 1], 3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(2), ])
        layers.extend(
            [nn.Conv2d(kernels[i + 1], 1, 3, padding=1, bias=False),
            nn.Sigmoid()])
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


if __name__ == "__main__":
    # Hyper Parameters
    in_ch = 4
    img_rows, img_cols = 224, 224
    for nb_layers in [1, 2, 3, 4]:
        discriminator = MLP(in_ch, nb_layers)
        patch_x = Variable(torch.FloatTensor(np.random.random(
            (1, in_ch, img_rows, img_cols))), volatile=True)
        logit = discriminator(patch_x)
        print("MLP(with {})->:".format(nb_layers))
        print("Network output", logit.shape)
        assert logit.shape[-2] == img_rows // (2**nb_layers)
        assert logit.shape[-1] == img_cols // (2**nb_layers)
