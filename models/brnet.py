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


class ConvUnit(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, is_bn, is_leaky, alpha):
        super(ConvUnit, self).__init__()
        # convolution block
        elems = []
        elems.append(nn.Conv2d(in_ch, out_ch, kernel))
        elems.append(nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True))
        if is_bn:
            elems.append(nn.BatchNorm2d(out_ch))
        self.block = nn.Sequential(*elems)

    def forward(self, x):
        x = self.block(x)
        return x


class UpPredict(nn.Module):
    def __init__(self, in_ch, out_ch, scale, is_bn, is_deconv, is_leaky, alpha):
        super(UpPredict, self).__init__()
        # upsampling and convolution block
        # H_out=(H−1)∗stride[0]−2∗padding[0]+kernel_size[0]+output_padding[0]
        elems = []
        if is_deconv:
            elems.append(nn.ConvTranspose2d(in_ch, out_ch, 2,
                                            stride=scale, output_padding=scale - 2))
        else:
            elems.append(nn.Upsample(scale_factor=scale))
            elems.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            elems.append(nn.LeakyReLU(alpha) if is_leaky else nn.ReLU(True))
        if is_bn:
            elems.append(nn.BatchNorm2d(out_ch))

        self.uppredict = nn.Sequential(*elems)

    def forward(self, x):
        x = self.uppredict(x)
        return x


class Backend(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,
                 is_deconv=False,
                 is_bn=False,
                 is_leaky=False,
                 alpha=0.1,):
        super(Backend, self).__init__()
        kernels = [x * base_kernel for x in [1, 2, 4, 8, 16]]
        # down&pooling
        self.downblock1 = UNetDownx2(
            nb_channel, kernels[0], is_bn, is_leaky, alpha)
        self.maxpool1 = nn.MaxPool2d(2)

        self.downblock2 = UNetDownx2(
            kernels[0], kernels[1], is_bn, is_leaky, alpha)
        self.maxpool2 = nn.MaxPool2d(2)

        self.downblock3 = UNetDownx2(
            kernels[1], kernels[2], is_bn, is_leaky, alpha)
        self.maxpool3 = nn.MaxPool2d(2)

        self.downblock4 = UNetDownx2(
            kernels[2], kernels[3], is_bn, is_leaky, alpha)
        self.maxpool4 = nn.MaxPool2d(2)

        # center convolution
        self.center = ConvBlock(kernels[3], kernels[4], is_bn, is_leaky, alpha)

        # up&concating
        self.upblock4 = UNetUpx2(
            kernels[4], kernels[3], is_deconv, is_bn, is_leaky, alpha)

        self.upblock3 = UNetUpx2(
            kernels[3], kernels[2], is_deconv, is_bn, is_leaky, alpha)

        self.upblock2 = UNetUpx2(
            kernels[2], kernels[1], is_deconv, is_bn, is_leaky, alpha)

        self.upblock1 = UNetUpx2(
            kernels[1], kernels[0], is_deconv, is_bn, is_leaky, alpha)

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
        ux3 = self.upblock3(ux4, dx31)
        ux2 = self.upblock2(ux3, dx21)
        ux1 = self.upblock1(ux2, dx11)

        return ux4, ux3, ux2, ux1


class BRNetv0(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,
                 is_bn=True,
                 is_leaky=True,):
        super(BRNetv0, self).__init__()
        is_deconv = False
        alpha = 0.1
        kernels = [x * base_kernel for x in [1, 2, 4, 8, 16]]
        self.backend = Backend(nb_channel,
                               nb_class,
                               base_kernel,
                               is_deconv,
                               is_bn,
                               is_leaky,
                               alpha)

        # generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(kernels[0], nb_class, 1),
            nn.Sigmoid() if nb_class == 1 else nn.Softmax(dim=1),)

        self.outconv2 = nn.Sequential(
            nn.Conv2d(kernels[0], nb_class, 1),
            nn.Sigmoid() if nb_class == 1 else nn.Softmax(dim=1),)

    def forward(self, x):
        ux4, ux3, ux2, ux1 = self.backend(x)
        out_1 = self.outconv1(ux1)
        out_2 = self.outconv2(ux1)
        return out_1, out_2


class BRNetv1(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,
                 is_bn=True,
                 is_leaky=True,
                 alpha=0.1,):
        super(BRNetv1, self).__init__()
        is_deconv = False
        kernels = [x * base_kernel for x in [1, 2, 4, 8, 16]]
        self.backend = Backend(nb_channel,
                               nb_class,
                               base_kernel,
                               is_deconv,
                               is_bn,
                               is_leaky,
                               alpha)

        # generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(kernels[0], nb_class, 1),
            nn.Sigmoid() if nb_class == 1 else nn.Softmax(dim=1),)

        self.outconv2 = nn.Sequential(
            nn.Conv2d(nb_class, nb_class, 3, padding=1),
            nn.Sigmoid() if nb_class == 1 else nn.Softmax(dim=1),)

    def forward(self, x):
        ux4, ux3, ux2, ux1 = self.backend(x)
        out_1 = self.outconv1(ux1)
        out_2 = self.outconv2(out_1)
        return out_1, out_2


class BRNetv2(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,
                 is_bn=True,
                 is_leaky=True,
                 alpha=0.1,):
        super(BRNetv2, self).__init__()
        is_deconv = False
        kernels = [x * base_kernel for x in [1, 2, 4, 8, 16]]
        self.backend = Backend(nb_channel,
                               nb_class,
                               base_kernel,
                               is_deconv,
                               is_bn,
                               is_leaky,
                               alpha)

        # generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(kernels[0], nb_class, 3, padding=1),
            nn.Sigmoid() if nb_class == 1 else nn.Softmax(dim=1),)

        self.outconv2 = nn.Sequential(
            nn.Conv2d(kernels[0] + nb_class, nb_class, 3, padding=1),
            nn.Sigmoid() if nb_class == 1 else nn.Softmax(dim=1),)

    def forward(self, x):
        ux4, ux3, ux2, ux1 = self.backend(x)

        seg = self.outconv1(ux1)
        feats = torch.cat([ux1, seg], dim=1)
        ol = self.outconv2(feats)
        return seg, ol


class BRNetv3(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,
                 is_bn=True,
                 is_leaky=True,
                 alpha=0.1,):
        super(BRNetv3, self).__init__()
        is_deconv = False
        kernels = [x * base_kernel for x in [1, 2, 4, 8, 16]]
        self.backend = Backend(nb_channel,
                               nb_class,
                               base_kernel,
                               is_deconv,
                               is_bn,
                               is_leaky,
                               alpha)

        self.outconv4 = ConvUnit(
            kernels[3], nb_class, 1, is_bn, is_leaky, alpha)
        self.outconv3 = ConvUnit(
            kernels[2], nb_class, 1, is_bn, is_leaky, alpha)
        self.outconv2 = ConvUnit(
            kernels[1], nb_class, 1, is_bn, is_leaky, alpha)
        self.outconv1 = ConvUnit(
            kernels[0], nb_class, 1, is_bn, is_leaky, alpha)

        if is_deconv:
            self.upscale_x2 = nn.ConvTranspose2d(
                nb_class, nb_class, 2, stride=2, output_padding=0)
            self.upscale_x4 = nn.ConvTranspose2d(
                nb_class, nb_class, 2, stride=4, output_padding=2)
            self.upscale_x8 = nn.ConvTranspose2d(
                nb_class, nb_class, 2, stride=8, output_padding=6)
        else:
            self.upscale_x2 = nn.Upsample(scale_factor=2)
            self.upscale_x4 = nn.Upsample(scale_factor=4)
            self.upscale_x8 = nn.Upsample(scale_factor=8)

        self.segmap = nn.Sequential(
            nn.Conv2d(4 * nb_class, nb_class, 3, padding=1),
            nn.Sigmoid() if nb_class == 1 else nn.Softmax(dim=1),)

        self.outline = nn.Sequential(
            nn.Conv2d(4 * nb_class, nb_class, 3, padding=1),
            nn.Sigmoid() if nb_class == 1 else nn.Softmax(dim=1),)

    def forward(self, x):
        ux4, ux3, ux2, ux1 = self.backend(x)
        out_4 = self.outconv4(ux4)
        out_4 = self.upscale_x8(out_4)

        out_3 = self.outconv3(ux3)
        out_3 = self.upscale_x4(out_3)

        out_2 = self.outconv2(ux2)
        out_2 = self.upscale_x2(out_2)

        out_1 = self.outconv1(ux1)

        feats = torch.cat([out_1, out_2, out_3, out_4], dim=1)
        seg = self.segmap(feats)
        ol = self.outline(feats)
        return seg, ol


class BRNetv4(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,
                 is_bn=True,
                 is_leaky=True,
                 alpha=0.1,):
        super(BRNetv4, self).__init__()
        is_deconv = False
        kernels = [x * base_kernel for x in [1, 2, 4, 8, 16]]
        self.backend = Backend(nb_channel,
                               nb_class,
                               base_kernel,
                               is_deconv,
                               is_bn,
                               is_leaky,
                               alpha)

        self.outconv4 = UpPredict(
            kernels[3], nb_class, 8, is_bn, is_deconv, is_leaky, alpha)
        self.outconv3 = UpPredict(
            kernels[2], nb_class, 4, is_bn, is_deconv, is_leaky, alpha)
        self.outconv2 = UpPredict(
            kernels[1], nb_class, 2, is_bn, is_deconv, is_leaky, alpha)

        self.outconv1 = nn.Sequential(
            nn.Conv2d(kernels[0], nb_class, 3, padding=1),
            nn.Sigmoid() if nb_class == 1 else nn.Softmax(dim=1),)

        self.side = nn.Sequential(
            nn.Conv2d(4 * nb_class, nb_class, 3, padding=1),
            nn.Sigmoid() if nb_class == 1 else nn.Softmax(dim=1),)

    def forward(self, x):
        ux4, ux3, ux2, ux1 = self.backend(x)

        out_4 = self.outconv4(ux4)
        out_3 = self.outconv3(ux3)
        out_2 = self.outconv2(ux2)
        out_1 = self.outconv1(ux1)

        feats = torch.cat([out_1, out_2, out_3, out_4], dim=1)
        out_side = self.side(feats)
        return out_1, out_side


if __name__ == "__main__":
    # Hyper Parameters
    nb_channel = 3
    nb_class = 1
    base_kernel = 24
    x = Variable(torch.FloatTensor(
        np.random.random((1, nb_channel, 224, 224))), volatile=True)

    generator = BRNetv0(nb_channel=3, nb_class=1)
    gen_y = generator(x)
    print("BRNetv0->:")
    print(" Network output 1", gen_y[0].shape)
    print(" Network output 2", gen_y[1].shape)

    generator = BRNetv1(nb_channel=3, nb_class=1)
    gen_y = generator(x)
    print("BRNetv1->:")
    print(" Network output 1", gen_y[0].shape)
    print(" Network output 2", gen_y[1].shape)

    generator = BRNetv2(nb_channel=3, nb_class=1)
    gen_y = generator(x)
    print("BRNetv2->:")
    print(" Network output 1", gen_y[0].shape)
    print(" Network output 2", gen_y[1].shape)

    generator = BRNetv3(nb_channel=3, nb_class=1)
    gen_y = generator(x)
    print("BRNetv3->:")
    print(" Network output 1", gen_y[0].shape)
    print(" Network output 2", gen_y[1].shape)

    generator = BRNetv4(nb_channel=3, nb_class=1)
    gen_y = generator(x)
    print("BRNetv4->:")
    print(" Network output 1", gen_y[0].shape)
    print(" Network output 2", gen_y[1].shape)
