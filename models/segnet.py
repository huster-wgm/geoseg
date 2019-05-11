#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import sys
sys.path.append('./models')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from blockunits import *
from torch.autograd import Variable


class SegNet(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,):
        super(SegNet, self).__init__()
        kernels = [x * base_kernel for x in [1, 2, 4, 8, 16]]
        # input conv
        self.inputconv = nn.Sequential(
            nn.Conv2d(nb_channel, kernels[0], 3, padding=1),)

        # down&pooling
        self.downblock1 = UNetDownx2(
            kernels[0], kernels[1], is_bn=True)
        self.maxpool1 = nn.MaxPool2d(2, return_indices=True)

        self.downblock2 = UNetDownx2(
            kernels[1], kernels[2], is_bn=True)
        self.maxpool2 = nn.MaxPool2d(2, return_indices=True)

        self.downblock3 = UNetDownx2(
            kernels[2], kernels[3], is_bn=True)
        self.maxpool3 = nn.MaxPool2d(2, return_indices=True)

        self.downblock4 = UNetDownx2(
            kernels[3], kernels[4], is_bn=True)
        self.maxpool4 = nn.MaxPool2d(2, return_indices=True)

        # up&unpooling
        self.upblock4 = SegNetUpx2(
            kernels[4], kernels[3], is_bn=True)

        self.upblock3 = SegNetUpx2(
            kernels[3], kernels[2], is_bn=True)

        self.upblock2 = SegNetUpx2(
            kernels[2], kernels[1], is_bn=True)

        self.upblock1 = SegNetUpx2(
            kernels[1], kernels[0], is_bn=True)

        # generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(kernels[0], nb_class, 1),
            nn.Sigmoid() if nb_class==1 else nn.Softmax(dim=1),)

    def forward(self, x):
        initconv = self.inputconv(x)

        # downconv
        dx11 = self.downblock1(initconv)
        dx12, max_1_indices = self.maxpool1(dx11)
        dx21 = self.downblock2(dx12)
        dx22, max_2_indices = self.maxpool2(dx21)
        dx31 = self.downblock3(dx22)
        dx32, max_3_indices = self.maxpool3(dx31)
        dx41 = self.downblock4(dx32)
        dx42, max_4_indices = self.maxpool4(dx41)

        # upconv
        ux4 = self.upblock4(dx42, max_4_indices, dx41.size())
        ux3 = self.upblock3(ux4, max_3_indices, dx31.size())
        ux2 = self.upblock2(ux3, max_2_indices, dx21.size())
        ux1 = self.upblock1(ux2, max_1_indices, dx11.size())

        return self.outconv1(ux1)


class SegNetvgg16(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,):
        super(SegNetvgg16, self).__init__()
        kernels = [x * base_kernel for x in [1, 2, 4, 8, 16]]
        # input conv
        self.inputconv = nn.Sequential(
            nn.Conv2d(nb_channel, kernels[0], 3, padding=1),)

        # down&pooling
        self.downblock1 = UNetDownx2(
            kernels[0], kernels[1], is_bn=True)
        self.maxpool1 = nn.MaxPool2d(2, return_indices=True)

        self.downblock2 = UNetDownx2(
            kernels[1], kernels[2], is_bn=True)
        self.maxpool2 = nn.MaxPool2d(2, return_indices=True)

        self.downblock3 = UNetDownx2(
            kernels[2], kernels[3], is_bn=True)
        self.maxpool3 = nn.MaxPool2d(2, return_indices=True)

        self.downblock4 = UNetDownx3(
            kernels[3], kernels[4], is_bn=True)
        self.maxpool4 = nn.MaxPool2d(2, return_indices=True)

        # up&unpooling
        self.upblock4 = SegNetUpx3(
            kernels[4], kernels[3], is_bn=True)

        self.upblock3 = SegNetUpx2(
            kernels[3], kernels[2], is_bn=True)

        self.upblock2 = SegNetUpx2(
            kernels[2], kernels[1], is_bn=True)

        self.upblock1 = SegNetUpx2(
            kernels[1], kernels[0], is_bn=True)

        # generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(kernels[0], nb_class, 1),
            nn.Sigmoid() if nb_class==1 else nn.Softmax(dim=1),)

    def forward(self, x):
        initconv = self.inputconv(x)
        # downconv
        dx11 = self.downblock1(initconv)
        dx12, max_1_indices = self.maxpool1(dx11)
        dx21 = self.downblock2(dx12)
        dx22, max_2_indices = self.maxpool2(dx21)
        dx31 = self.downblock3(dx22)
        dx32, max_3_indices = self.maxpool3(dx31)
        dx41 = self.downblock4(dx32)
        dx42, max_4_indices = self.maxpool4(dx41)

        # upconv
        ux4 = self.upblock4(dx42, max_4_indices, dx41.size())
        ux3 = self.upblock3(ux4, max_3_indices, dx31.size())
        ux2 = self.upblock2(ux3, max_2_indices, dx21.size())
        ux1 = self.upblock1(ux2, max_1_indices, dx11.size())

        return self.outconv1(ux1)



if __name__ == "__main__":
    # Hyper Parameters
    nb_channel = 3
    nb_class = 1
    base_kernel = 24
    x = Variable(torch.FloatTensor(
        np.random.random((1, nb_channel, 224, 224))), volatile=True)

    generator = SegNet(nb_channel, nb_class, base_kernel)
    gen_y = generator(x)
    print("SegNet->:")
    print(" Network output ", gen_y.shape)

    generator = SegNetvgg16(nb_channel, nb_class, base_kernel)
    gen_y = generator(x)
    print("SegNetvgg16->:")
    print(" Network output ", gen_y.shape)
