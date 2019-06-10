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


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)


class SFCNS(object):
    """ SFCN(Stacked Fully Convolutional Networks)
        Wu, G., Guo, Y., Song, X., Guo, Z., Zhang, H., Shi, X., ... & Shao, X. (2019). \
        A Stacked Fully Convolutional Networks with Feature Alignment Framework for Multi-Label Land-cover Segmentation. \
        Remote Sensing, 11(9), 1051.
        Basic models should have single output(e.g. FCNs, UNet, FPN etc)
    """

    def __init__(self, args):
        from models.fcn import FCN8s, FCN16s, FCN32s
        from models.unet import UNet
        from models.segnet import SegNet
        from models.resunet import ResUNet
        from models.fpn import FPN
        assert len(args.models) >= 2, "Ensembled models should be more than 2."
        self.names = args.models
        self.models = []
        params = []
        for name in self.names:
            model = eval(name)(args.src_ch, args.tar_ch, args.base_kernel)
            if args.cuda:
                model.cuda()
            self.models.append(model)
            params.append({'params': model.parameters()})
        self.optimizer = optim.Adam(params, lr=args.lr, betas=optim_betas)
        self.symbol = "SFCNx{}-".format(len(self.names)) + "-".join(self.names)


class BFCNS(object):
    """ BFCN(Boosted Fully Convolutional Networks)
        
        Basic models should have single output(e.g. FCNs, UNet, FPN etc)
    """

    def __init__(self, args):
        from models.fcn import FCN8s, FCN16s, FCN32s
        from models.unet import UNet
        from models.segnet import SegNet
        from models.resunet import ResUNet
        from models.fpn import FPN
        assert len(args.models) >= 2, "Ensembled models should be more than 2."
        self.names = args.models
        self.models = []
        self.optimizers = []
        for name in self.names:
            model = eval(name)(args.src_ch, args.tar_ch, args.base_kernel)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=optim_betas)
            if args.cuda:
                model.cuda()
            self.models.append(model)
            self.optimizers.append(optimizer)
        self.symbol = "BFCNx{}-".format(len(self.names)) + "-".join(self.names)


def load_model(args):
    if args.ensemble == "stacking":
        net = SFCNS(args)
    else:
        ValueError(
            "Ensemble method [{}] is not supported yet.".format(args.ensemble))
    return net


def load_dataset(args):
    from datasets import LS
    train_set = LS(args.dataset, "train")
    val_set = LS(args.dataset, "val")
    return train_set, val_set


def set_trainer(args, method):
    from esrunner import stackTrainer
    if args.ensemble == "stacking":
        args.alpha = 1.0
        return stackTrainer(args, method, is_multi=False)
    else:
        ValueError(
            "Ensemble method [{}] is not supported yet.".format(args.ensemble))


def main(args):
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")

    # initialize datasets
    train_set, val_set = load_dataset(args)
    print("Dataset : {} ==> Train : {} ; Val : {} .".format(args.dataset, len(train_set), len(val_set)))

    # initialize network
    args.src_ch = train_set.src_ch
    args.tar_ch = train_set.tar_ch
    net = load_model(args)
    print("Model : {} ==> Src_ch : {} ; Tar_ch : {} .".format(net.symbol, args.src_ch, args.tar_ch))

    # initialize runner
    method = "{}-{}".format(net.symbol, args.dataset) 
    run = set_trainer(args, method)
    print("Start training ...")

    run.training(net, [train_set, val_set])
    run.save_log()
    run.learning_curve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-ensemble', type=str, default='stacking', choices=['bagging', 'stacking', 'boosting'],
                        help='method of ensemble learning ')
    parser.add_argument('-models', nargs='+', type=str, default=["FCN8s", "UNet", "FPN"],
                        help='models used for ensembling ')
    parser.add_argument('-dataset', type=str, default='Vaihingen', choices=['Vaihingen', 'PotsdamIRRG', 'PotsdamRGB', 'PotsdamRGBIR'],
                        help='dataset for training models')
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
                        help='learning rate for discriminator')
    parser.add_argument('-cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()
    optim_betas = (0.9, 0.999)
    main(args)
