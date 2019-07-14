#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""

import os
import torch
import argparse
import numpy as np
import torch.optim as optim
from datasets import load_dataset

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)


def load_model(args):
    from models.fcn import FCN8s, FCN16s, FCN32s
    from models.unet import UNet
    from models.segnet import SegNet
    from models.resunet import ResUNet
    from models.fpn import FPN
    from models.mcfcn import MCFCN
    from models.brnet import BRNet, BRNetv1, BRNetv2, BRNetv3, BRNetv4
    
    net = eval(args.net)(args.src_ch, args.tar_ch, args.base_kernel)
    if args.cuda:
        net.cuda()
    net.optimizer = optim.Adam(
        net.parameters(), lr=args.lr, betas=optim_betas)
    return net


def set_trainer(args, method):
    from runner import Trainer, mcTrainer, brTrainer
    if "MCFCN" in args.net:
        return mcTrainer(args, method, is_multi=True)
    elif "BRNet" in args.net:
        args.alpha = 0.5
        return brTrainer(args, method, is_multi=True)
    else:
        return Trainer(args, method, is_multi=False)


def main(args):
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")

    # initialize datasets
    if "MCFCN" in args.net:
        mode = 'IMS'
    elif "BRNet" in args.net:
        mode = 'IME'
    else:
        mode = 'IM'
    train_set, val_set = load_dataset(args.root, mode)
    print("Dataset : {} ==> Train : {} ; Val : {}".format(args.root, len(train_set), len(val_set)))

    # initialize network
    args.src_ch = train_set.src_ch
    args.tar_ch = train_set.tar_ch
    net = load_model(args)
    print("Model : {} ==> (Src_ch : {} ; Tar_ch : {} ; Base_Kernel : {})".format(args.net, args.src_ch, args.tar_ch, args.base_kernel))

    # initialize runner
    method = "{}-{}*{}*{}-{}".format(args.net, args.src_ch, args.tar_ch, args.base_kernel, args.root) 
    run = set_trainer(args, method)
    print("Start training ...")

    run.training(net, [train_set, val_set])
    run.save_log()
    run.learning_curve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-root', type=str, default='NZ32km2', 
                        help='root dir of dataset for training models')
    parser.add_argument('-net', type=str, default='FCN8s',
                        help='network type for training')
    parser.add_argument('-base_kernel', type=int, default=24,
                        help='base number of kernels')
    parser.add_argument('-trigger', type=str, default='epoch', choices=['epoch', 'iter'],
                        help='trigger type for logging')
    parser.add_argument('-interval', type=int, default=10,
                        help='interval for logging')
    parser.add_argument('-terminal', type=int, default=100,
                        help='terminal for training ')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='batch_size for training ')
    parser.add_argument('-lr', type=float, default=1e-4,
                        help='learning rate for optimization')
    parser.add_argument('-cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()
    optim_betas = (0.9, 0.999)
    main(args)
