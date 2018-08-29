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


class LinkNetUp(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, is_leaky=False):
        super(LinkNetUp, self).__init__()
        alpha = 0.1
        self.upsample = None
        # B, 2C, H, W -> B, C/2, H, W
        self.convbn1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch//2),)

        if stride == 2:
            # B, C/2, H, W -> B, C/2, 2*H, 2*W
            self.upsample = nn.ConvTranspose2d(out_ch//2, out_ch//2, kernel_size=3, stride=2, padding=1)
            self.upbn = nn.BatchNorm2d(out_ch//2)

        # B, C/2, H, W -> B, C, H, W
        self.convbn2 = nn.Sequential(
            nn.Conv2d(out_ch//2, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),)

        self.relu = nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True)

    def forward(self, x):
        x = self.convbn1(x)
        x = self.relu(x)
        if self.upsample is not None:
            required_size = [2*i for i in x.shape[2:]]
            x = self.upsample(x, output_size=required_size)
            x = self.upbn(x)
            x = self.relu(x)
        x = self.convbn2(x)
        return self.relu(x)


class LinkNet(nn.Module):
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
        super(LinkNet, self).__init__()
        # Currently hardcoded for ResNet-18
        kernels = [base_kernel * i for i in [1, 2, 4, 8]]
        layers = [2, 2, 2, 2]
        alpha = 0.1
        # inital input channel
        self.in_ch = kernels[0]

        # intial block
        self.inconv = nn.Sequential(
            nn.Conv2d(nb_channel, kernels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(kernels[0]),
            nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoder
        block = ResBasicBlock
        self.encoder1 = self._make_layer(block, kernels[0], layers[0], stride=1, is_leaky=is_leaky)
        self.encoder2 = self._make_layer(block, kernels[1], layers[1], stride=2, is_leaky=is_leaky)
        self.encoder3 = self._make_layer(block, kernels[2], layers[2], stride=2, is_leaky=is_leaky)
        self.encoder4 = self._make_layer(block, kernels[3], layers[3], stride=2, is_leaky=is_leaky)

        # Decoder
        self.decoder4 = LinkNetUp(kernels[3], kernels[2], stride=2, is_leaky=is_leaky)
        self.decoder3 = LinkNetUp(kernels[2], kernels[1], stride=2, is_leaky=is_leaky)
        self.decoder2 = LinkNetUp(kernels[1], kernels[0], stride=2, is_leaky=is_leaky)
        self.decoder1 = LinkNetUp(kernels[0], kernels[0], stride=1, is_leaky=is_leaky)

        # final block
        self.deconvup = nn.ConvTranspose2d(kernels[0], base_kernel//2, kernel_size=3, stride=2, padding=1)
        self.deconvbn = nn.BatchNorm2d(base_kernel//2)

        self.convbn1 = nn.Sequential(
            nn.Conv2d(base_kernel//2, base_kernel//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_kernel//2),)

        self.devconvout = nn.Sequential(
            nn.ConvTranspose2d(base_kernel//2, nb_class, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid() if nb_class==1 else nn.Softmax(dim=1),)

        self.relu = nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True)


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
        # inital block
        x = self.inconv(x)
        x = self.maxpool(x)

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with skip addition
        d4 = self.decoder4(e4)
        d4 += e3
        d3 = self.decoder3(d4)
        d3 += e2
        d2 = self.decoder2(d3)
        d2 += e1
        d1 = self.decoder1(d2)

        # Final Classification
        required_size = [2*i for i in d1.shape[2:]]
        f1 = self.deconvup(d1, output_size=required_size)
        f1 = self.deconvbn(f1)
        f1 = self.relu(f1)
        f2 = self.convbn1(f1)
        f2 = self.relu(f2)
        f3 = self.devconvout(f2)

        return f3



if __name__ == "__main__":
    # Hyper Parameters
    x = Variable(torch.FloatTensor(np.random.random((1, 3, 224, 224))), volatile=True)

    generator = LinkNet(nb_channel=3, nb_class=1, base_kernel=64)
    gen_y = generator(x)
    print("LinkNet->:")
    print(" Network output ", gen_y.shape)
