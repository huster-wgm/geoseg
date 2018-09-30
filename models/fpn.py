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
from blockunits import *
from torch.autograd import Variable


class FPN(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,):
        super(FPN, self).__init__()
        kernels = [x * base_kernel for x in [1, 2, 4, 8, 16]]
        # down&pooling
        self.downblock1 = UNetDownx2(
            nb_channel, kernels[0], is_bn=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.downblock2 = UNetDownx2(
            kernels[0], kernels[1], is_bn=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.downblock3 = UNetDownx2(
            kernels[1], kernels[2], is_bn=True)
        self.maxpool3 = nn.MaxPool2d(2)

        self.downblock4 = UNetDownx2(
            kernels[2], kernels[3], is_bn=True)
        self.maxpool4 = nn.MaxPool2d(2)

        # center convolution
        self.center = ConvBlock(kernels[3], kernels[4], is_bn=True)

        # up&concating
        self.upblock4 = UNetUpx2(
            kernels[4], kernels[3], is_bn=True)
        self.sideconv4 = nn.Sequential(
            nn.Conv2d(kernels[3], nb_class, 1),
            nn.ReLU(True),)

        self.upblock3 = UNetUpx2(
            kernels[3], kernels[2], is_bn=True)
        self.sideconv3 = nn.Sequential(
            nn.Conv2d(kernels[2], nb_class, 1),
            nn.ReLU(True),)

        self.upblock2 = UNetUpx2(
            kernels[2], kernels[1], is_bn=True)
        self.sideconv2 = nn.Sequential(
            nn.Conv2d(kernels[1], nb_class, 1),
            nn.ReLU(True),)

        self.upblock1 = UNetUpx2(
            kernels[1], kernels[0], is_bn=True)
        self.sideconv1 = nn.Sequential(
            nn.Conv2d(kernels[0], nb_class, 1),
            nn.ReLU(True),)

        # generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(4*nb_class, nb_class, 3, padding=1),
            nn.Sigmoid() if nb_class==1 else nn.Softmax(dim=1),)


    def forward(self, x):
        dx11 = self.downblock1(x)
        dx12 = self.maxpool1(dx11)

        dx21 = self.downblock2(dx12)
        dx22 = self.maxpool2(dx21)

        dx31 = self.downblock3(dx22)
        dx32 = self.maxpool3(dx31)

        dx41 = self.downblock4(dx32)
        dx42 = self.maxpool4(dx41)

        cx = self.center(dx42)

        ux4 = self.upblock4(cx, dx41)
        out_4 = self.sideconv4(ux4)
        up4 = F.upsample(out_4, scale_factor=8, mode='bilinear')

        ux3 = self.upblock3(ux4, dx31)
        out_3 = self.sideconv3(ux3)
        up3 = F.upsample(out_3, scale_factor=4, mode='bilinear')

        ux2 = self.upblock2(ux3, dx21)
        out_2 = self.sideconv2(ux2)
        up2 = F.upsample(out_2, scale_factor=2, mode='bilinear')

        ux1 = self.upblock1(ux2, dx11)
        out_1 = self.sideconv1(ux1)

        cat = torch.cat([up4, up3, up2, out_1], dim=1)
        return self.outconv1(cat)


class FPNvgg16(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,):
        super(FPNvgg16, self).__init__()
        kernels = [x * base_kernel for x in [1, 2, 4, 8, 16]]
        # down&pooling
        self.downblock1 = UNetDownx2(
            nb_channel, kernels[0], is_bn=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.downblock2 = UNetDownx2(
            kernels[0], kernels[1], is_bn=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.downblock3 = UNetDownx2(
            kernels[1], kernels[2], is_bn=True)
        self.maxpool3 = nn.MaxPool2d(2)

        self.downblock4 = UNetDownx3(
            kernels[2], kernels[3], is_bn=True)
        self.maxpool4 = nn.MaxPool2d(2)

        # center convolution
        self.center = UNetDownx3(kernels[3], kernels[4], is_bn=True)

        # up&concating
        self.upblock4 = UNetUpx3(
            kernels[4], kernels[3], is_bn=True)
        self.sideconv4 = nn.Sequential(
            nn.Conv2d(kernels[3], nb_class, 1),
            nn.ReLU(True),)

        self.upblock3 = UNetUpx2(
            kernels[3], kernels[2], is_bn=True)
        self.sideconv3 = nn.Sequential(
            nn.Conv2d(kernels[2], nb_class, 1),
            nn.ReLU(True),)

        self.upblock2 = UNetUpx2(
            kernels[2], kernels[1], is_bn=True)
        self.sideconv2 = nn.Sequential(
            nn.Conv2d(kernels[1], nb_class, 1),
            nn.ReLU(True),)

        self.upblock1 = UNetUpx2(
            kernels[1], kernels[0], is_bn=True)
        self.sideconv1 = nn.Sequential(
            nn.Conv2d(kernels[0], nb_class, 1),
            nn.ReLU(True),)

        # generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(4*nb_class, nb_class, 3, padding=1),
            nn.Sigmoid() if nb_class==1 else nn.Softmax(dim=1),)

    def forward(self, x):
        dx11 = self.downblock1(x)
        dx12 = self.maxpool1(dx11)

        dx21 = self.downblock2(dx12)
        dx22 = self.maxpool2(dx21)

        dx31 = self.downblock3(dx22)
        dx32 = self.maxpool3(dx31)

        dx41 = self.downblock4(dx32)
        dx42 = self.maxpool4(dx41)

        cx = self.center(dx42)

        ux4 = self.upblock4(cx, dx41)
        out_4 = self.sideconv4(ux4)
        up4 = F.upsample(out_4, scale_factor=8, mode='bilinear')

        ux3 = self.upblock3(ux4, dx31)
        out_3 = self.sideconv3(ux3)
        up3 = F.upsample(out_3, scale_factor=4, mode='bilinear')

        ux2 = self.upblock2(ux3, dx21)
        out_2 = self.sideconv2(ux2)
        up2 = F.upsample(out_2, scale_factor=2, mode='bilinear')

        ux1 = self.upblock1(ux2, dx11)
        out_1 = self.sideconv1(ux1)

        cat = torch.cat([up4, up3, up2, out_1], dim=1)
        return self.outconv1(cat)


if __name__ == "__main__":
    # Hyper Parameters
    nb_channel = 3
    nb_class = 1
    base_kernel = 24
    x = Variable(torch.FloatTensor(
        np.random.random((1, nb_channel, 224, 224))), volatile=True)

    generator = FPN(nb_channel, nb_class, base_kernel)
    gen_y = generator(x)
    print("FPN->:")
    print(" Network output ", gen_y.shape)

    generator = FPNvgg16(nb_channel, nb_class, base_kernel)
    gen_y = generator(x)
    print("FPNvgg16->:")
    print(" Network output ", gen_y.shape)
