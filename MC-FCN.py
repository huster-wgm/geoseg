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

from models import mcfcn
from utils.runner import mcTrainer
from config import LSSubdataset


class MCFCN(object):
    """
        packed generator and discriminator in patchGAN
        input and output are paired for next step classification
    """

    def __init__(self, args):
        self.model = mcfcn.MCFCN(args.in_ch, args.out_ch, args.base_kernel)


def main(args):
    method = os.path.basename(__file__).split(".")[0]

    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")

    # initialize datasets
    datasets = LSSubdataset()

    args.in_ch = datasets.in_ch
    args.out_ch = datasets.out_ch

    # initialize network
    net = MCFCN(args)
    if args.cuda:
        net.model.cuda()
    net.optimizer = optim.Adam(
        net.model.parameters(), lr=args.lr, betas=optim_betas)

    # initialize runner
    run = mcTrainer(args, method, is_multi=True)

    print("Start training {}...".format(method))
    run.training(net, [datasets.train8xsub, datasets.val])
    run.save_log()
    run.learning_curve()

    run.evaluating(net.model, datasets.train, 'train')
    run.evaluating(net.model, datasets.val, 'val')
    run.evaluating(net.model, datasets.test, "test")

    run.save_checkpoint(net.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-alpha', type=float, default=0.5,
                        help='alpha ratiot between main and sub loss ')
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
