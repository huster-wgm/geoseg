#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 2 19:29:18 2017

@author: Go-hiroaki

network models in pytorch
"""
import sys
sys.path.append('./models')
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(ConvBlock, self).__init__()
        # convolution block
        if is_bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),)
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)

    def forward(self, x):
        x = self.block(x)
        return x


class UNetDownx2(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(UNetDownx2, self).__init__()
        # convolution block
        if is_bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),)
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)

    def forward(self, x):
        x = self.block(x)
        return x


class UNetDownx3(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False, is_leaky=False, alpha=0.1):
        super(UNetDownx3, self).__init__()
        # convolution block
        if is_bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.BatchNorm2d(out_ch),)
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),)

    def forward(self, x):
        x = self.block(x)
        return x


class UNetUpx2(nn.Module):
    def __init__(self, in_ch, out_ch, is_deconv=False, is_bn=False, is_leaky=False, alpha=0.1):
        super(UNetUpx2, self).__init__()
        # upsampling and convolution block
        if is_deconv:
            self.upscale = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        else:
            if is_bn:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.BatchNorm2d(out_ch),
                    nn.Upsample(scale_factor=2),)
            else:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.Upsample(scale_factor=2),)
        self.block = UNetDownx2(in_ch, out_ch, is_bn, is_leaky, alpha)

    def forward(self, x1, x2):
        x1 = self.upscale(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.block(x)
        return x


class UNetUpx3(nn.Module):
    def __init__(self, in_ch, out_ch,is_deconv=False, is_bn=False, is_leaky=False, alpha=0.1):
        super(UNetUpx3, self).__init__()
        # upsampling and convolution block
        if is_deconv:
            self.upscale = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        else:
            if is_bn:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.BatchNorm2d(out_ch),
                    nn.Upsample(scale_factor=2),)
            else:
                self.upscale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=1),
                    nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True),
                    nn.Upsample(scale_factor=2),)
        self.block = UNetDownx3(in_ch, out_ch, is_bn, is_leaky, alpha)

    def forward(self, x1, x2):
        x1 = self.upscale(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.block(x)
        return x


class SegNetUpx2(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False):
        super(SegNetUpx2, self).__init__()
        # upsampling and convolution block
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.block = UNetDownx2(in_ch, out_ch, is_bn)

    def forward(self, x, indices, output_shape):
        x = self.unpool(x, indices, output_shape)
        x = self.block(x)
        return x


class SegNetUpx3(nn.Module):
    def __init__(self, in_ch, out_ch, is_bn=False):
        super(SegNetUpx3, self).__init__()
        # upsampling and convolution block
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.block = UNetDownx3(in_ch, out_ch, is_bn)

    def forward(self, x, indices, output_shape):
        x = self.unpool(x, indices, output_shape)
        x = self.block(x)
        return x


def conv3x3bn(in_ch, out_ch, stride=1):
    "3x3 convolution with padding"
    convbn = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3,
                  stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),)
    return convbn


class ResBasicBlock(nn.Module):
    expansion = 1
    # modified from original pytorch implementation
    # see http://pytorch.org/docs/0.2.0/_modules/torchvision/models/resnet.html

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, is_leaky=False):
        super(ResBasicBlock, self).__init__()
        alpha = 0.1
        self.conv1bn = conv3x3bn(in_ch, out_ch, stride)
        self.relu = nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True)
        self.conv2bn = conv3x3bn(out_ch, out_ch)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1bn(x)
        out = self.relu(out)

        out = self.conv2bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResBottleneck(nn.Module):
    expansion = 4
    # modified from original pytorch implementation
    # see http://pytorch.org/docs/0.2.0/_modules/torchvision/models/resnet.html

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, is_leaky=False):
        super(ResBottleneck, self).__init__()
        alpha = 0.1
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * 4)
        self.relu = nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
