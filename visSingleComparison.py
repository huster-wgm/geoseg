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
import numpy as np
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
    Multi-house comparison using different methods
      args:
        .checkpoints: pretrained pytorch model
        .target: target output [segmap, edge]
        .img_nb: number of image to be display at once
        .gen_nb: number of image to be generated
        .dataset: dataset for prediction
    """
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")
    if not os.path.exists(os.path.join(Result_DIR, 'single-comparison')):
        os.mkdir(os.path.join(Result_DIR, 'single-comparison'))
    # setup dataset
    dataset = rs2018TrainDataset(args.dataset, args.img_rows, args.img_cols, split='all', origin='vector')
    data_loader = DataLoader(dataset, args.disp_cols, num_workers=4, shuffle=False)
    batch_iterator = iter(data_loader)
    # setup visualization parameter
    spaces = args.spaces
    direction = args.direction
    disp_cols = args.disp_cols
    disp_rows = len(args.checkpoints) + 1

    if args.eval_fn == 'ov':
        evaluate = metrics.overall_accuracy
    elif args.eval_fn == 'precision':
        evaluate = metrics.precision
    elif args.eval_fn == 'recall':
        evaluate = metrics.recall
    elif args.eval_fn == 'f1_score':
        evaluate = metrics.f1_score
    elif args.eval_fn == 'jaccard':
        evaluate = metrics.jaccard
    elif args.eval_fn == 'kappa':
        evaluate = metrics.kappa
    else:
        raise ValueError(
            'Evaluation function of {} is not support yet.'.format(args.eval_fn))

    selected_pairs = []
    steps = len(dataset) // args.disp_cols
    # generate prediction
    for step in range(steps):
        pairs = []
        x, y = next(batch_iterator)
        # generate original image
        x_imgs = []
        for i in range(y.shape[0]):
            x_img = vision.tensor_to_img(x[i], False)
            x_imgs.append(vision.add_barrier(x_img, spaces))
        pairs.append(x_imgs)

        # convert tensor to Variable
        x = Variable(x, volatile=True)
        if args.cuda:
            x = x.cuda()

        pair_perfoms = []
        # generate prediction by checkpoint
        for checkpoint in args.checkpoints:
            # load model
            model = load_checkpoint(checkpoint)
            if args.cuda:
                model.cuda()
            model.eval()

            # generate prediction
            if checkpoint.startswith("MC") or checkpoint.startswith("mt"):
                y_pred = model(x)[0]
            else:
                y_pred = model(x)

            if args.cuda:
                y_pred = y_pred.data.cpu()
            else:
                y_pred = y_pred.data

            # generate rgb prediction
            pred_imgs, perfoms = [], []
            for j in range(y.shape[0]):
                # calculate performance
                perfoms.append(evaluate(y_pred[j], y[j]))
                # generate prediction image
                pred_img = vision.tensor_to_img(y_pred[j], True)[:, :, 0]
                y_img = vision.tensor_to_img(y[j], False)[:, :, 0]
                rgb_img = vision.pair_to_rgb(pred_img, y_img, args.color)
                if args.target == 'edge':
                    # extract edges from segmentation map
                    pred_img = img_to_edge(pred_img)
                    y_img = img_to_edge(y_img)
                    rgb_img = vision.pair_to_rgb(pred_img, y_img, args.color, use_dilation=True, disk_value=args.disk)
                pred_imgs.append(vision.add_barrier(rgb_img, spaces))

            pairs.append(pred_imgs)
            pair_perfoms.append(perfoms)

        pair_perfoms = np.array(pair_perfoms)
        # select pair of prediction by significance
        for i in range(pair_perfoms.shape[1]):
            status = False
            for j in range(pair_perfoms.shape[0] - 1):
                value = pair_perfoms[j, i]
                next_value = pair_perfoms[j + 1, i]
                if (next_value - value) >= args.significance:
                    status = True
            if status:
                selected_pairs += [pair[i] for pair in pairs]

        # break loop with enough number
        if len(selected_pairs) >= args.gen_nb * disp_rows * disp_cols:
            print("Required number : {} ; Collected number : {}.".format(args.gen_nb * disp_rows * disp_cols, len(selected_pairs)))
            break

    for nb in range(args.gen_nb):
        selected_lices = selected_pairs[nb * disp_rows *
                                        disp_cols: (nb + 1) * disp_rows * disp_cols]
        sort_slices = []
        for k in range(disp_rows):
            sort_slices += selected_lices[k::disp_rows]

        result_img = vision.slices_to_img(sort_slices, [disp_rows, disp_cols])
        result_img = vision.add_color_bar(result_img, spaces[1], 'white')
        name = "{}_{}_{}.png".format(
            args.target, '_'.join([x.split('_')[0] for x in args.checkpoints]), nb)
        imsave(os.path.join(Result_DIR, 'single-comparison', name),
               result_img)
        print("Saving {} ...".format(name))


if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-checkpoints', nargs='+', type=str, default=["FCN_sample.pth", "UNet_sample.pth"],
                        help='checkpoints used for making prediction ')
    parser.add_argument('-spaces', nargs='+', type=int, default=[2, 5],
                        help='barrier spaces for merging ')
    parser.add_argument('-direction', type=str, default="horizontal", choices=['horizontal', 'vertical'],
                        help='merge image direction ')
    parser.add_argument('-disp_cols', type=int, default=8,
                        help='cols for displaying image ')
    parser.add_argument('-target', type=str, default="segmap", choices=['segmap', 'edge'],
                        help='target for model prediction [segmap, edge]')
    parser.add_argument('-edge_fn', type=str, default='canny', choices=['shift', 'canny'],
                        help='method used for edge extraction')
    parser.add_argument('-gen_nb', type=int, default=2,
                        help='number of generated image ')
    parser.add_argument('-eval_fn', type=str, default='kappa', choices=['ov', 'precision', 'recall', 'f1_score', 'jaccard', 'kappa'],
                        help='method used for evaluate performance')
    parser.add_argument('-significance', type=float, default=0.00,
                        help='significant different level between methods ')
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
    parser.add_argument('-batch_size', type=int, default=32,
                        help='batch size for model prediction ')
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
