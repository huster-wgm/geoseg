#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @CreateTime:   2018-01-26T16:50:00+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import gc
import argparse
import torch
import itertools
import numpy as np
from torch.autograd import Variable
from skimage.io import imsave, imread

from utils import vision
from utils.runner import load_checkpoint

Data_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
Result_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')


def get_pred(model, x, cuda=True):
    y_pred = model(x)
    if cuda:
        y_pred = y_pred.data.cpu().numpy()
    else:
        y_pred = y_pred.data.numpy()
    return y_pred


def get_mc_pred(model, x, cuda=True):
    y_pred = model(x)[0]
    if cuda:
        y_pred = y_pred.data.cpu().numpy()
    else:
        y_pred = y_pred.data.numpy()
    return y_pred


def get_mt_pred(model, x, cuda=True):
    y_pred, y_pred_edge = model(x)
    if cuda:
        y_pred = y_pred.data.cpu().numpy()
        y_pred_edge = y_pred_edge.data.cpu().numpy()
    else:
        y_pred = y_pred.data.numpy()
        y_pred_edge = y_pred_edge.data.numpy()
    return y_pred, y_pred_edge


def main(args):
    """
    Multi-house comparison using different methods
      args:
        .checkpoints: pretrained pytorch model
        .data: data path for prediction
    """
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")
    if not os.path.exists(os.path.join(Result_DIR, 'area-binary')):
        os.makedirs(os.path.join(Result_DIR, 'area-binary'))
    # setup edge extraction function
    if args.edge_fn == 'shift':
        img_to_edge = vision.shift_edge
    elif args.edge_fn == 'canny':
        img_to_edge = vision.canny_edge
    else:
        raise ValueError(
            'Edge extraction method of {} is not support yet.'.format(args.edge_fn))
    # read and align image
    src_img = imread(args.data)
    x_slices, x_shapes = vision.img_to_slices(
        src_img, args.img_rows, args.img_cols)
    x_slices = np.array(x_slices)

    for checkpoint in args.checkpoints:
        # load models
        model = load_checkpoint(checkpoint)
        if args.cuda:
            model.cuda()
        model.eval()
        # predict by batch
        y_preds, y_pred_edges = [], []
        steps = x_slices.shape[0] // args.batch_size
        for step in range(steps + 1):
            print("Predicting by {} at {}/{} \r".format(checkpoint, step, steps))
            if step < steps:
                x = x_slices[step *
                             args.batch_size:(step + 1) * args.batch_size]
            else:
                x = x_slices[step * args.batch_size:]
            x = (x / 255).transpose((0, 3, 1, 2)).astype('float32')
            x = Variable(torch.FloatTensor(x), volatile=True)
            if args.cuda:
                x = x.cuda()
            # generate prediction
            y_pred_edge = []
            if checkpoint.startswith("MC"):
                y_pred = get_mc_pred(model, x, args.cuda)
            elif checkpoint.startswith("mt"):
                y_pred, y_pred_edge = get_mt_pred(model, x, args.cuda)
            else:
                y_pred = get_pred(model, x, args.cuda)
            y_preds.append(y_pred)
            y_pred_edges.append(y_pred_edge)
        y_preds = np.concatenate(y_preds)
        y_pred_edges = np.concatenate(y_pred_edges)
        del x_slices

        results = []
        for i in range(y_preds.shape[0]):
            pred_img = vision.array_to_img(y_preds[i], True)[:, :, 0]
            if checkpoint.startswith("mt"):
                pred_edge = vision.array_to_img(
                    y_pred_edges[i], True)[:, :, 0]
            if args.target == 'edge':
                # extract edges from segmentation map
                pred_img = img_to_edge(pred_img)
                # if checkpoint.startswith("mt"):
                #     pred_img = pred_edge
            results.append(pred_img)
        del y_preds, y_pred_edges

        # merge slices into image
        result_img = vision.slices_to_img(results, x_shapes)
        del results
        # TODO: add metadata to .tif file
        name = "{}_area_{}.tif".format(
            checkpoint.split('_')[0], args.target)
        imsave(os.path.join(Result_DIR, 'area-binary', name), result_img)
        print("Saving {} ...".format(name))


if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-checkpoints', nargs='+', type=str, default=["FCN_sample.pth"],
                        help='checkpoints used for making prediction ')
    parser.add_argument('-target', type=str, default="segmap", choices=['segmap', 'edge'],
                        help='target for model prediction [segmap, edge]')
    parser.add_argument('-edge_fn', type=str, default='shift', choices=['shift', 'canny'],
                        help='method used for edge extraction')
    parser.add_argument('-data', type=str, default="RS-2018/test",
                        help='data dir for processing')
    parser.add_argument('-img_rows', type=int, default=224,
                        help='img rows for croping ')
    parser.add_argument('-img_cols', type=int, default=224,
                        help='img cols for croping ')
    parser.add_argument('-batch_size', type=int, default=24,
                        help='batch size for model prediction ')
    parser.add_argument('-cuda', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='using cuda for optimization')
    args = parser.parse_args()

    main(args)
