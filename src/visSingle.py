#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import argparse
import os
import time
import torch
import metrics
import vision
import numpy as np
import pandas as pd

from datasets import load_dataset
from skimage.io import imread, imsave
from torch.utils.data import DataLoader


DIR = os.path.dirname(os.path.abspath(__file__))
Result_DIR = os.path.join(DIR, '../result/')
Checkpoint_DIR = os.path.join(DIR, '../checkpoint')


def load_checkpoint(checkpoint, cuda):
    from models.fcn import FCN8s, FCN16s, FCN32s
    from models.unet import UNet
    from models.segnet import SegNet
    from models.resunet import ResUNet
    from models.fpn import FPN
    from models.mcfcn import MCFCN
    from models.brnet import BRNet, BRNetv2
    assert os.path.exists("{}/{}".format(Checkpoint_DIR, checkpoint)
                          ), "{} does not exists.".format(checkpoint)
    method = checkpoint.split('_')[0]
    model = method.split('-')[0]
    is_multi = False
    if "MCFCN" in model or "BRNet" in model:
        is_multi = True
    src_ch, tar_ch, base_kernel = [int(x) for x in method.split('-')[1].split("*")]
    net = eval(model)(src_ch, tar_ch, base_kernel)
    net.load_state_dict(
        torch.load(os.path.join(Checkpoint_DIR, checkpoint)))
    if cuda:
        net.cuda()
    print("Loaded checkpoint: {}".format(checkpoint))
    return net.eval(), is_multi


def main(args):
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")

    evaluators = [metrics.OAAcc(), metrics.Precision(), metrics.Recall(), 
                  metrics.F1Score(), metrics.Kappa(), metrics.Jaccard()]

    for checkpoint in args.checkpoints:
        print("Handling by {} ...\r".format(checkpoint))
        Save_DIR = os.path.join(Result_DIR, 'single', checkpoint.split("_")[0])
        if not os.path.exists(Save_DIR):
            os.makedirs(Save_DIR)
        # initialize datasets
        infos = checkpoint.split('_')[0].split('-')
        _, valset = load_dataset(infos[2], "IM")
        print("Testing with {}-Dataset: {} examples".format(infos[2], len(valset)))
        # Load checkpoint
        model, is_multi = load_checkpoint(checkpoint, args.cuda)
        # load data
        data_loader = DataLoader(valset, 1, num_workers=4,
                                 shuffle=False, pin_memory=True,)
        performs = [[] for i in range(len(evaluators))]
        imgsets = []
        with torch.set_grad_enabled(False):
            for idx, sample in enumerate(data_loader):
                # get tensors from sample
                x = sample["src"]
                y = sample["tar"]
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                if is_multi:
                    gen_y = model(x)[0]
                else:
                    gen_y = model(x)
                # get performance
                for i, evaluator in enumerate(evaluators):
                    performs[i].append(evaluator(gen_y.detach(), y.detach())[0].item())
                if args.cuda:
                    x = x.detach().cpu()
                    y = x.detach().cpu()
                    gen_y = gen_y.detach().cpu()
                x = x.numpy()[0].transpose((1, 2, 0))
                y = y.numpy()[0].transpose((1, 2, 0))
                gen_y = gen_y.numpy()[0].transpose((1, 2, 0))
                x_img = valset._src2img(x, whitespace=False)
                y_img = valset._tar2img(y, whitespace=False)
                gen_img = valset._tar2img(gen_y, whitespace=False)
                canny_x = vision.canny_edge(x_img)
                canny_y = vision.canny_edge(y_img)
                canny_gen = vision.canny_edge(gen_img)
                # mask_pair = vision.pair_to_rgb(gen_img, y_img, args.color)
                canny_pair = vision.pair_to_rgb(canny_y, canny_x, args.color, use_dilation=True, disk_value=args.disk)
                edge_pair = vision.pair_to_rgb(canny_gen, canny_y, args.color, use_dilation=True, disk_value=args.disk)
                imgsets.append([vision.add_barrier(x_img, args.spaces),
                                vision.add_barrier(canny_pair, args.spaces),
                                # vision.add_barrier(mask_pair, args.spaces),
                                vision.add_barrier(edge_pair, args.spaces),
                               ])
                if len(imgsets) >= args.disp_cols * args.gen_nb:
                    break
            # visualization
            for i in range(args.gen_nb):
                imgset = []
                for j in range(args.disp_cols):
                    imgset.append(np.concatenate(imgsets[i*args.disp_cols+j], axis=0))
                vis_img = np.concatenate(imgset, axis=1)
                name = "{}_canny_segmap_edge_{}.png".format(
                    checkpoint.split('_')[0], i)
                imsave(os.path.join(Save_DIR, name),
                       vision.add_barrier(vis_img, args.spaces))
                print("Saving {} ...".format(name))


if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-checkpoints', nargs='+', type=str, default=["FCN_sample.pth"],
                        help='checkpoints used for making prediction ')
    parser.add_argument('-spaces', nargs='+', type=int, default=[2, 5],
                        help='barrier space for merging ')
    parser.add_argument('-disp_cols', type=int, default=8,
                        help='cols for displaying image ')
    parser.add_argument('-gen_nb', type=int, default=5,
                        help='number of generated image ')
    parser.add_argument('-color', type=str, default='white',
                        help='background color for generated rgb result ')
    parser.add_argument('-disk', type=int, default=2,
                        help='dilation level ')
    parser.add_argument('-cuda', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='using cuda for optimization')
    args = parser.parse_args()

    main(args)
