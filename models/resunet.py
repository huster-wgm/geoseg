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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from blockunits import *
from torch.autograd import Variable


class ResUNet(nn.Module):
    """
     modified from pytorch-semseg implementation
     see https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/linknet.py
     original[torch7] https://codeac29.github.io/projects/linknet/
    """
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,
                 is_leaky=False,):
        super(ResUNet, self).__init__()
        # Currently hardcoded for ResNet-18
        kernels = [base_kernel * i for i in [1, 2, 4, 8, 16]]
        layers = [2, 2, 2, 2]

        # inital input channel
        self.in_ch = kernels[0]

        # intial block
        self.inconv = nn.Sequential(
            nn.Conv2d(nb_channel, kernels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(kernels[0]),
            nn.LeakyReLU(0.1) if is_leaky else nn.ReLU(True),
            nn.Conv2d(kernels[0], kernels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(kernels[0]),
            nn.LeakyReLU(0.1) if is_leaky else nn.ReLU(True),)

        # Encoder
        block = ResBasicBlock
        self.encoder1 = self._make_layer(block, kernels[1], layers[0], stride=2, is_leaky=is_leaky)
        self.encoder2 = self._make_layer(block, kernels[2], layers[1], stride=2, is_leaky=is_leaky)
        self.encoder3 = self._make_layer(block, kernels[3], layers[2], stride=2, is_leaky=is_leaky)
        self.encoder4 = self._make_layer(block, kernels[4], layers[3], stride=2, is_leaky=is_leaky)

        # up&concating
        self.decoder4 = UNetUpx2(
            kernels[4], kernels[3], True, True, is_leaky)
        self.decoder3 = UNetUpx2(
            kernels[3], kernels[2], True, True, is_leaky)
        self.decoder2 = UNetUpx2(
            kernels[2], kernels[1], True, True, is_leaky)
        self.decoder1 = UNetUpx2(
            kernels[1], kernels[0], True, True, is_leaky)

        # generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(kernels[0], nb_class, 1),
            nn.Sigmoid() if nb_class==1 else nn.Softmax(dim=1),)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_ch, blocks, stride=1, is_leaky=False):
        downsample = None
        if stride != 1 or self.in_ch != out_ch * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * block.expansion),
            )

        layers = []
        layers.append(block(self.in_ch, out_ch, stride, downsample, is_leaky))
        self.in_ch = out_ch * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_ch, out_ch, is_leaky=is_leaky))

        return nn.Sequential(*layers)


    def forward(self, x):
        # conv
        x = self.inconv(x)

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with skip concanate
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, x)

        return self.outconv1(d1)


if __name__ == "__main__":
    # Hyper Parameters
    nb_channel = 3
    nb_class = 1
    base_kernel = 24
    x = Variable(torch.FloatTensor(
        np.random.random((1, nb_channel, 224, 224))), volatile=True)

    generator = ResUNet(nb_channel, nb_class, base_kernel)
    gen_y = generator(x)
    print("ResUNet->:")
    print(" Network output ", gen_y.shape)
