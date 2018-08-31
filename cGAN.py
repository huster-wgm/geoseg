#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @CreateTime:   2018-01-02T20:57:09+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import argparse
import torch
import torch.optim as optim

from models import resunet, mlp
from utils.runner import cganTrainer
from config import LSdataset, LSEdataset


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


class PatchGAN(object):
    """
        packed generator and discriminator in patchGAN
        input and output are paired for next step classification
    """

    def __init__(self, args):
        self.generator = resunet.ResUNet(
            args.in_ch, args.out_ch, args.base_kernel, args.is_leaky)
        self.discriminator = mlp.MLP(
            args.in_ch + args.out_ch, args.patch_layers, args.base_kernel)
        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)


def main(args):
    method = os.path.basename(__file__).split(".")[0]
    if args.is_leaky:
        method += '-leaky'
    method += '-p' + str(args.patch_layers)

    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")

    # initialize datasets
    datasets = LSdataset()

    args.in_ch = datasets.in_ch
    args.out_ch = datasets.out_ch

    # initialize network
    net = PatchGAN(args)
    if args.cuda:
        net.generator.cuda()
        net.discriminator.cuda()
    net.g_optimizer = optim.Adam(
        net.generator.parameters(), lr=args.g_lr, betas=optim_betas)
    net.d_optimizer = optim.Adam(
        net.discriminator.parameters(), lr=args.d_lr, betas=optim_betas)

    # initialize runner
    run = cganTrainer(args, method, is_multi=False)

    print("Start training {}...".format(method))
    run.training(net, [datasets.train, datasets.val])
    run.save_log()
    run.learning_curve()

    run.evaluating(net.generator, datasets.train, 'train')
    run.evaluating(net.generator, datasets.val, 'val')
    run.evaluating(net.generator, datasets.test, "test")

    run.save_checkpoint(net.generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-is_leaky', type=bool, default=False,
                        help='used leakyReLU or not')
    parser.add_argument('-lamb', type=float, default=10,
                        help='lambda weight for segmentation loss')
    parser.add_argument('-trigger', type=str, default='epoch', choices=['epoch', 'iter'],
                        help='trigger type for logging')
    parser.add_argument('-interval', type=int, default=1,
                        help='interval for logging')
    parser.add_argument('-terminal', type=int, default=100,
                        help='terminal for training ')
    parser.add_argument('-base_kernel', type=int, default=24,
                        help='base number of kernels')
    parser.add_argument('-patch_layers', type=int, default=3,
                        help='patch_layers for training ')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='batch_size for training ')
    parser.add_argument('-d_lr', type=float, default=1e-4,
                        help='learning rate for discriminator')
    parser.add_argument('-g_lr', type=float, default=1e-3,
                        help='learning rate for generator')
    parser.add_argument('-cuda', type=bool, default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()
    optim_betas = (0.9, 0.999)
    main(args)
