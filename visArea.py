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
import warnings
import time
import argparse
import torch
import itertools
import numpy as np
import pandas as pd
from torch.autograd import Variable

from utils import vision
from utils import metrics
from utils.runner import load_checkpoint
from torch.utils import data
from torch.utils.data import DataLoader
from skimage.io import imread
from skimage.external.tifffile import imsave
from skimage.transform import resize

Dataset_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'dataset')
Result_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')

metrics_name = ['overall_accuracy', 'precision',
                'recall', 'f1-score', 'jac', 'kappa']


class areaDataset(data.Dataset):
    """Image Dataset Object for land and segmentation images
    args:
        dataset: (str) root of the dataset e.g. 'RS-2018/test'
        split: (str) part of the data ['train', 'val', 'all']
        require_sub: (bool) return subsampling ground truth
        sub: (int) subsampling level, default 8
        origin: (str) croping method from origin data [slice, vector]
    """

    def __init__(self, dataset, img_rows, img_cols,
                 data_range=[0.0, 1.0],):
        self.dataset = os.path.join(Dataset_DIR, dataset, 'slice')
        self._landpath = os.path.join(self.dataset, 'land', '%s')
        self._segpath = os.path.join(self.dataset, 'segmap', '%s')
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.require_resize = False
        # get dataset config
        config = pd.read_csv(os.path.join(self.dataset, 'statistic.csv'))
        ori_rows = config['img_rows'][0]
        ori_cols = config['img_cols'][0]
        if ori_rows != self.img_rows or ori_cols != self.img_cols:
            warnings.warn(
                'Required size: is not consistent with original size.')
            self.require_resize = True
        self.nb_rows = config['nb_rows'][0]
        self.nb_cols = config['nb_cols'][0]
        # get image ids
        infos = pd.read_csv(os.path.join(
            self.dataset, 'all-infos.csv'))
        assert infos.shape[0] == self.nb_rows * \
            self.nb_cols, 'Image number should be consistent.'
        id_min, id_max = int(data_range[0] * self.nb_rows) * self.nb_cols, \
            int(data_range[1] * self.nb_rows) * self.nb_cols
        self.shapes = [int(data_range[1] * self.nb_rows) -
                       int(data_range[0] * self.nb_rows), self.nb_cols]
        infos = infos.iloc[id_min:id_max, :]
        self.ids = infos['id'].tolist()

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # load land image
        img_land = imread(self._landpath % img_id)
        if self.require_resize:
            img_land = resize(
                img_land, (self.img_rows, self.img_cols), mode='edge')
            img_land = img_land.astype('float32').transpose((2, 0, 1))
        else:
            img_land = (img_land / 255).astype('float32').transpose((2, 0, 1))
        # load segmap
        img_seg = imread(self._segpath % img_id)
        if self.require_resize:
            img_seg = resize(
                img_seg, (self.img_rows, self.img_cols), mode='edge')
            img_seg = img_seg.astype('float32').transpose((2, 0, 1))
        else:
            img_seg = (np.expand_dims(img_seg, axis=-1) /
                       255).astype('float32').transpose((2, 0, 1))
        return img_land, img_seg

    def __len__(self):
        return len(self.ids)


def get_pred(model, args):
    # setup dataset
    dataset = areaDataset(args.dataset, args.img_rows,
                          args.img_cols, args.row_range)
    data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                             shuffle=False)
    batch_iterator = iter(data_loader)
    shapes = dataset.shapes
    # predict by batch
    y_preds, y_gts = [], []
    steps = len(dataset) // args.batch_size
    if steps * args.batch_size < len(dataset):
        steps += 1
    for step in range(steps):
        x, y = next(batch_iterator)
        x = Variable(x, volatile=True)
        if args.cuda:
            x = x.cuda()
        # generate prediction
        y_pred = model(x)
        if args.cuda:
            y_pred = y_pred.data.cpu()
        else:
            y_pred = y_pred.data
        y_preds.append(y_pred)
        y_gts.append(y)
    y_preds = torch.cat(y_preds, 0)
    y_gts = torch.cat(y_gts, 0)
    assert y_preds.shape[0] == len(dataset), "All data should be iterated."
    return y_preds, y_gts, shapes


def get_mc_mt_pred(model, args):
    # setup dataset
    dataset = areaDataset(args.dataset, args.img_rows,
                          args.img_cols, args.row_range)
    data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                             shuffle=False)
    batch_iterator = iter(data_loader)
    shapes = dataset.shapes
    # predict by batch
    y_preds, y_gts = [], []
    steps = len(dataset) // args.batch_size
    if steps * args.batch_size < len(dataset):
        steps += 1
    for step in range(steps):
        x, y = next(batch_iterator)
        x = Variable(x, volatile=True)
        if args.cuda:
            x = x.cuda()
        # generate prediction
        y_pred = model(x)[0]
        if args.cuda:
            y_pred = y_pred.data.cpu()
        else:
            y_pred = y_pred.data
        y_preds.append(y_pred)
        y_gts.append(y)
    y_preds = torch.cat(y_preds, 0)
    y_gts = torch.cat(y_gts, 0)
    assert y_preds.shape[0] == len(dataset), "All data should be iterated."
    return y_preds, y_gts, shapes


def pred_to_img(y_preds, y_gts):
    # convert predict tensor to img & get rgb result
    rgb_results = []
    for i in range(y_preds.shape[0]):
        pred_img = vision.tensor_to_img(y_preds[i], True)[:, :, 0]
        y_img = vision.tensor_to_img(y_gts[i], False)[:, :, 0]
        rgb_img = vision.pair_to_rgb(pred_img, y_img, args.color)
        if args.target == 'edge':
            # extract edges from segmentation map
            pred_img = img_to_edge(pred_img)
            y_img = img_to_edge(y_img)
            # enhance outline border
            rgb_img = vision.pair_to_rgb(
                pred_img, y_img, args.color, use_dilation=True, disk_value=args.disk)
        rgb_results.append(rgb_img)
    return rgb_results


def main(args):
    """
    Multi-house comparison using different methods
      args:
        .checkpoints: pretrained pytorch model
        .target: target output [segmap, edge]
        .data: data dir for prediction
    """
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")
    if not os.path.exists(os.path.join(Result_DIR, 'area')):
        os.makedirs(os.path.join(Result_DIR, 'area'))

    for checkpoint in args.checkpoints:
        # Load model
        model = load_checkpoint(checkpoint)
        if args.cuda:
            model.cuda()
        model.eval()
        oa, precision, recall, f1, jac, kappa = 0, 0, 0, 0, 0, 0
        # slice by number of parts to save memory
        for idx in range(args.parts):
            print("Predicting by {} at part-{}/{} \r".format(checkpoint,
                                                             idx + 1, args.parts))
            # selected part of the data
            row_range = [(1.0 / args.parts) * idx,
                         (1.0 / args.parts) * (1 + idx)]
            if idx == args.parts - 1:
                row_range = [(1.0 / args.parts) * idx, 1.0]
            args.row_range = [round(val, 1) for val in row_range]
            # get prediction
            if checkpoint.startswith("MC") or checkpoint.startswith("mt"):
                y_preds, y_gts, shapes = get_mc_mt_pred(model, args)
            else:
                y_preds, y_gts, shapes = get_pred(model, args)

            # evaluate performance
            oa += metrics.overall_accuracy(y_preds, y_gts)
            precision += metrics.precision(y_preds, y_gts)
            recall += metrics.recall(y_preds, y_gts)
            f1 += metrics.f1_score(y_preds, y_gts)
            jac += metrics.jaccard(y_preds, y_gts)
            kappa += metrics.kappa(y_preds, y_gts)

            # merge slices into image & save result image
            rgb_results = pred_to_img(y_preds, y_gts)
            del y_preds, y_gts
            result_img = vision.slices_to_img(rgb_results, shapes)
            del rgb_results
            name = "{}_area_{}_part_{}.tif".format(
                checkpoint.split('_')[0], args.target, idx + 1)
            imsave(os.path.join(Result_DIR, 'area', name),
                   result_img,  compress=6)
            print("Saving {} ...".format(name))
            del result_img

            gc.collect()

        perform = ["%0.3f" % (value / args.parts)
                   for value in [oa, precision, recall, f1, jac, kappa]]

        perform = pd.DataFrame([[time.strftime("%h%d_%H"), checkpoint, args.target] + perform],
                               columns=['time', 'checkpoint', 'type'] + metrics_name)

        # save performance
        log_path = os.path.join(
            Result_DIR, "area-performs.csv".format(args.target))
        if os.path.exists(log_path):
            performs = pd.read_csv(log_path)
        else:
            performs = pd.DataFrame([])
        performs = performs.append(perform, ignore_index=True)
        performs.to_csv(log_path, index=False)


if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-checkpoints', nargs='+', type=str, default=["FCN_sample.pth"],
                        help='checkpoints used for making prediction ')
    parser.add_argument('-target', type=str, default="segmap", choices=['segmap', 'edge'],
                        help='target for model prediction [segmap, edge]')
    parser.add_argument('-edge_fn', type=str, default='canny', choices=['shift', 'canny'],
                        help='method used for edge extraction')
    parser.add_argument('-dataset', type=str, default="RS-2018/test",
                        help='dataset path for loading ')
    parser.add_argument('-color', type=str, default='white', choices=['white', 'black'],
                        help='background color for generated rgb result ')
    parser.add_argument('-parts', type=int, default=5,
                        help='number of parts to divided whole area')
    parser.add_argument('-disk', type=int, default=2,
                        help='dilation level ')
    parser.add_argument('-img_rows', type=int, default=224,
                        help='img rows for croping ')
    parser.add_argument('-img_cols', type=int, default=224,
                        help='img cols for croping ')
    parser.add_argument('-batch_size', type=int, default=24,
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