#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *


class VGGbackend(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 base_kernel=64):
        super(VGGbackend, self).__init__()
        self.nb_channel = nb_channel
        kernels = [base_kernel * i for i in [1, 2, 4, 8, 16]]

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.nb_channel, kernels[0], 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[0], kernels[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(kernels[0], kernels[1], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[1], kernels[1], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(kernels[1], kernels[2], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[2], kernels[2], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[2], kernels[2], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(kernels[2], kernels[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[3], kernels[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[3], kernels[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),)

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(kernels[3], kernels[4], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[4], kernels[4], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels[4], kernels[4], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),)

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        return conv5, conv4, conv3


class FCN32s(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,):
        super(FCN32s, self).__init__()
        kernels = [base_kernel * i for i in [1, 2, 4, 8, 16]]

        self.backend = VGGbackend(nb_channel, base_kernel)

        self.classifier = nn.Sequential(
            nn.Conv2d(kernels[4], 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, nb_class, 1),)

        # generate output
        self.outconv1 = nn.Sigmoid() if nb_class == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        conv5, conv4, conv3 = self.backend(x)

        score = self.classifier(conv5)
        # up = F.upsample(score, x.size()[2:], mode='bilinear')
        up = F.interpolate(score, x.size()[2:], mode='bilinear')
        return self.outconv1(up)


class FCN16s(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,
                 zks=3,):
        super(FCN16s, self).__init__()
        kernels = [base_kernel * i for i in [1, 2, 4, 8, 16]]

        self.backend = VGGbackend(nb_channel, base_kernel)

        self.classifier = nn.Sequential(
            nn.Conv2d(kernels[4], 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, nb_class, 1),)

        self.score_pool4 = nn.Conv2d(kernels[3], nb_class, 1)

        # generate output
        self.outconv1 = nn.Sigmoid() if nb_class == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        conv5, conv4, conv3 = self.backend(x)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)

        # score = F.upsample(score, score_pool4.size()[2:], mode='bilinear')
        score = F.interpolate(score, score_pool4.size()[2:], mode='bilinear')
        score += score_pool4
        # up = F.upsample(score, x.size()[2:], mode='bilinear')
        up = F.interpolate(score, x.size()[2:], mode='bilinear')
        return self.outconv1(up)


class FCN8s(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,
                 zks=3,):
        super(FCN8s, self).__init__()
        kernels = [base_kernel * i for i in [1, 2, 4, 8, 16]]

        self.backend = VGGbackend(nb_channel, base_kernel)

        self.classifier = nn.Sequential(
            nn.Conv2d(kernels[4], 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, nb_class, 1),)

        self.score_pool4 = nn.Conv2d(kernels[3], nb_class, 1)
        self.score_pool3 = nn.Conv2d(kernels[2], nb_class, 1)

        # generate output
        self.outconv1 = nn.Sigmoid() if nb_class == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        conv5, conv4, conv3 = self.backend(x)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)

        # score = F.upsample(score, score_pool4.size()[2:], mode='bilinear')
        score = F.interpolate(score, score_pool4.size()[2:], mode='bilinear') 
        score += score_pool4
        # score = F.upsample(score, score_pool3.size()[2:], mode='bilinear')
        score = F.interpolate(score, score_pool3.size()[2:], mode='bilinear')
        score += score_pool3
        # up = F.upsample(score, x.size()[2:], mode='bilinear')
        up = F.interpolate(score, x.size()[2:], mode='bilinear')
        return self.outconv1(up)


if __name__ == "__main__":
    # Hyper Parameters
    nb_channel = 3
    nb_class = 1
    base_kernel = 24
    x = torch.FloatTensor(
        np.random.random((1, nb_channel, 224, 224)))

    generator = FCN32s(nb_channel, nb_class, base_kernel)
    total_params = sum(p.numel() for p in generator.parameters())
    gen_y = generator(x)
    print("FCN32s->:")
    print(" Params: {:0.1f}M".format(total_params / (10**6)))
    print(" Network output: ", gen_y.shape)


    generator = FCN16s(nb_channel, nb_class, base_kernel)
    total_params = sum(p.numel() for p in generator.parameters())
    gen_y = generator(x)
    print("FCN16s->:")
    print(" Params: {:0.1f}M".format(total_params / (10**6)))
    print(" Network output: ", gen_y.shape)

    generator = FCN8s(nb_channel, nb_class, base_kernel)
    total_params = sum(p.numel() for p in generator.parameters())
    gen_y = generator(x)
    print("FCN8s->:")
    print(" Params: {:0.1f}M".format(total_params / (10**6)))
    print(" Network output: ", gen_y.shape)