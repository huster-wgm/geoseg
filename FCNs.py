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

from models import fcn
from utils.runner import Trainer
from config import LSdataset


class FCNs(object):
    """
        packed generator and discriminator in patchGAN
        input and output are paired for next step classification
    """

    def __init__(self, args):
        if args.ver == 'FCN8s':
            self.model = fcn.FCN8s(args.in_ch, args.out_ch, args.base_kernel)
        elif args.ver == 'FCN16s':
            self.model = fcn.FCN16s(args.in_ch, args.out_ch, args.base_kernel)
        elif args.ver == 'FCN32s':
            self.model = fcn.FCN32s(args.in_ch, args.out_ch, args.base_kernel)
        else:
            raise ValueError(
                "Version should be in['FCN8s', 'FCN16s','FCN32s']")


def main(args):
    method = args.ver
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")

    # initialize datasets
    datasets = LSdataset()

    args.in_ch = datasets.in_ch
    args.out_ch = datasets.out_ch

    # initialize network
    net = FCNs(args)
    if args.cuda:
        net.model.cuda()
    net.optimizer = optim.Adam(
        net.model.parameters(), lr=args.lr, betas=optim_betas)

    # initialize runner
    run = Trainer(args, method, is_multi=False)

    print("Start training {}...".format(method))
    run.training(net, [datasets.train, datasets.val])
    run.save_log()
    run.learning_curve()

    run.evaluating(net.model, datasets.train, 'train')
    run.evaluating(net.model, datasets.val, 'val')
    run.evaluating(net.model, datasets.test, "test")

    run.save_checkpoint(net.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-ver', type=str, default="FCN8s",
                        help='Version of Fully convolutional Networks ')
    parser.add_argument('-trigger', type=str, default='epoch', choices=['epoch', 'iter'],
                        help='trigger type for logging')
    parser.add_argument('-interval', type=int, default=10,
                        help='interval for logging')
    parser.add_argument('-terminal', type=int, default=100,
                        help='terminal for training ')
    parser.add_argument('-base_kernel', type=int, default=24,
                        help='base number of kernels')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='batch_size for training ')
    parser.add_argument('-lr', type=float, default=2e-4,
                        help='learning rate for discriminator')
    parser.add_argument('-cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()
    optim_betas = (0.9, 0.999)
    main(args)
