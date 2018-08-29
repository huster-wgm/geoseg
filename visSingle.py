#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @CreateTime:   2018-01-26T16:50:00+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import argparse
import torch
from torch.autograd import Variable
from skimage.io import imsave

from utils import vision
from utils import metrics
from utils.runner import load_checkpoint
from utils.datasets import rs2018TrainDataset
from torch.utils.data import DataLoader
from skimage.color import rgb2gray

Result_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')


def main(args):
    """
    Single house level prediction using different methods
      args:
        .checkpoints: pretrained pytorch model
        .img_nb: number of image to be display at once
        .gen_nb: number of image to be generated
        .dataset: dataset for prediction
    """
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")
    if not os.path.exists(os.path.join(Result_DIR, 'single')):
        os.mkdir(os.path.join(Result_DIR, 'single'))
    # setup dataset
    dataset = rs2018TrainDataset(args.dataset, args.img_rows,
                        args.img_cols, split='all', origin='vector')
    data_loader = DataLoader(dataset, args.disp_cols,
                             num_workers=4, shuffle=False)
    batch_iterator = iter(data_loader)
    # setup visualization parameter
    spaces = args.spaces
    direction = args.direction
    disp_cols = args.disp_cols
    disp_rows = 3 + 1

    for idx in range(args.gen_nb):
        # load data
        x, y = next(batch_iterator)

        # generate original image
        x_imgs = []
        for i in range(y.shape[0]):
            x_img = vision.tensor_to_img(x[i], False)
            x_imgs.append(vision.add_barrier(x_img, spaces))

        # generate canny outline
        canny_imgs = []
        for i in range(y.shape[0]):
            x_img = vision.tensor_to_img(x[i], False)
            x_img = rgb2gray(x_img)
            canny_img = img_to_edge(x_img)
            y_img = vision.tensor_to_img(y[i], False)[:, :, 0]
            y_img = img_to_edge(y_img)
            canny_rgb = vision.pair_to_rgb(
                canny_img, y_img, args.color, use_dilation=True, disk_value=args.disk)
            canny_imgs.append(vision.add_barrier(canny_rgb, spaces))

        x = Variable(x, volatile=True)
        if args.cuda:
            x = x.cuda()

        # make prediction via checkpoints
        for checkpoint in args.checkpoints:
            model = load_checkpoint(checkpoint)
            if args.cuda:
                model.cuda()
            model.eval()

            results = []

            # forward generator
            if checkpoint.startswith("MC") or checkpoint.startswith("mt"):
                y_pred = model(x)[0]
            else:
                y_pred = model(x)

            if args.cuda:
                y_pred = y_pred.data.cpu()
            else:
                y_pred = y_pred.data

            # generate segmentation map
            seg_imgs = []
            for j in range(y.shape[0]):
                pred_img = vision.tensor_to_img(y_pred[j], True)[:, :, 0]
                y_img = vision.tensor_to_img(y[j], False)[:, :, 0]
                img_rgb = vision.pair_to_rgb(pred_img, y_img, args.color)
                seg_imgs.append(vision.add_barrier(img_rgb, spaces))

            # generate edges
            edge_imgs = []
            for k in range(y.shape[0]):
                pred_img = vision.tensor_to_img(y_pred[k], True)[:, :, 0]
                y_img = vision.tensor_to_img(y[k], False)[:, :, 0]
                # extract edges from segmentation map
                pred_img = img_to_edge(pred_img)
                y_img = img_to_edge(y_img)
                img_rgb = vision.pair_to_rgb(
                    pred_img, y_img, args.color, use_dilation=True, disk_value=args.disk)
                edge_imgs.append(vision.add_barrier(img_rgb, spaces))

            # add original images
            results += x_imgs
            # add canny images
            results += canny_imgs
            # add segmentation
            results += seg_imgs
            # add edge
            results += edge_imgs

            result_img = vision.slices_to_img(results, [disp_rows, disp_cols])
            name = "{}_canny_segmap_edge_{}.png".format(
                checkpoint.split('_')[0], idx)
            imsave(os.path.join(Result_DIR, 'single', name),
                   vision.add_color_bar(result_img, spaces[1], 'white'))
            print("Saving {} ...".format(name))


if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-checkpoints', nargs='+', type=str, default=["FCN_sample.pth"],
                        help='checkpoints used for making prediction ')
    parser.add_argument('-spaces', nargs='+', type=int, default=[2, 5],
                        help='barrier space for merging ')
    parser.add_argument('-direction', type=str, default="horizontal", choices=['horizontal', 'vertical'],
                        help='merge image direction ')
    parser.add_argument('-disp_cols', type=int, default=8,
                        help='cols for displaying image ')
    parser.add_argument('-edge_fn', type=str, default='canny', choices=['shift', 'canny'],
                        help='method used for edge extraction')
    parser.add_argument('-gen_nb', type=int, default=1,
                        help='number of generated image ')
    parser.add_argument('-color', type=str, default='white',
                        help='background color for generated rgb result ')
    parser.add_argument('-dataset', type=str, default="RS-2018/test",
                        help='dataset path for loading ')
    parser.add_argument('-disk', type=int, default=2,
                        help='dilation level ')
    parser.add_argument('-img_rows', type=int, default=224,
                        help='img rows for croping ')
    parser.add_argument('-img_cols', type=int, default=224,
                        help='img cols for croping ')
    parser.add_argument('-cuda', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='using cuda for optimization')
    args = parser.parse_args()

    # setup edge extraction function
    if args.edge_fn == 'shift':
        img_to_edge = vision.shift_edge
    elif args.edge_fn == 'canny':
        img_to_edge = vision.canny_edge
    else:
        raise ValueError(
            'Edge extraction method of {} is not support yet.'.format(args.edge_fn))

    main(args)
