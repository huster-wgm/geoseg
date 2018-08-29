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
import time
import argparse
import warnings
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
from skimage.io import imread, imsave
from skimage.transform import resize

Dataset_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'dataset')
Result_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')

corners = ['top-left', 'top-right', 'centroid', 'bottom-left', 'bottom-right']
metrics_name = ['overall_accuracy', 'precision',
                'recall', 'f1-score', 'jac', 'kappa']


class cornerDataset(data.Dataset):
    """Image Dataset Object for land and segmentation images
    args:
        dataset: (str) root of the dataset e.g. 'RS-2018/train'
        split: (str) part of the data ['train', 'val', 'all']
        require_sub: (bool) return subsampling ground truth
        sub: (int) subsampling level, default 8
        origin: (str) croping method from origin data [slice, vector]
    """

    def __init__(self, dataset, img_rows, img_cols, corner, buffer, sizes):
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
        row_range, col_range = self._get_corner_range(corner, sizes, buffer)
        self.ids, self.shapes = self._get_idx_by_range(row_range, col_range)

    def _get_corner_range(self, corner, sizes, buffer):
        """
        get position range of top-left, top-right, centroid, bottom-left and bottom-right
        args:
          corner: (str) target corner top-left, top-right, centroid, bottom-left and bottom-right
          sizes: (list) [rows, cols] target size of rows and cols for displaying
          buffer: (int) images from boundries to ignore
        return position range
        """
        # set corners
        row_top = [buffer, buffer + sizes[0]]
        row_bottom = [self.nb_rows - buffer - sizes[0], self.nb_rows - buffer]
        row_centroid = [self.nb_rows // 2 - sizes[0] // 2,
                        self.nb_rows // 2 - sizes[0] // 2 + sizes[0]]
        col_left = [buffer, buffer + sizes[1]]
        col_right = [self.nb_cols - buffer - sizes[1], self.nb_cols - buffer]
        col_centroid = [self.nb_cols // 2 - sizes[1] // 2,
                        self.nb_cols // 2 - sizes[1] // 2 + sizes[1]]

        if corner == corners[0]:
            pos = [row_top, col_left]
        elif corner == corners[1]:
            pos = [row_top, col_right]
        elif corner == corners[2]:
            pos = [row_centroid, col_centroid]
        elif corner == corners[3]:
            pos = [row_bottom, col_left]
        elif corner == corners[4]:
            pos = [row_bottom, col_right]
        else:
            raise ValueError(
                "Required corner-{} is not supperted".format(corner))
        return pos

    def _get_idx_by_range(self, row_range, col_range):
        """
        get idx of img slices from selected area
        args:
          row_range: (list) range for rows
          col_range: (list) range for cols
          img_shapes: (list) nb_rows, nb_cols for img_slices
        return selected idx, shapes
        """
        assert row_range[0] < row_range[1], "row_range, max should larger than min"
        assert col_range[0] < col_range[1], "col_range, max should larger than min"
        assert row_range[1] <= self.nb_rows, "max range of row should less that nb_rows"
        assert col_range[1] <= self.nb_cols, "max range of col should less that nb_cols"
        selected_idx = []
        row_rg = range(row_range[0], row_range[1], 1)
        col_rg = range(col_range[0], col_range[1], 1)
        for row, col in itertools.product(row_rg, col_rg):
            file = 'img_{}.png'.format(row * self.nb_cols + col)
            selected_idx.append(file)

        selected_shapes = [row_range[1] -
                           row_range[0], col_range[1] - col_range[0]]
        return selected_idx, selected_shapes

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
    dataset = cornerDataset(args.dataset, args.img_rows,
                            args.img_cols, args.corner, args.buffer, args.sizes)
    data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                             shuffle=False)
    batch_iterator = iter(data_loader)
    shapes = dataset.shapes
    # predict by batch
    x_array, y_preds, y_gts = [], [], []
    steps = len(dataset) // args.batch_size
    if steps * args.batch_size < len(dataset):
        steps += 1
    for step in range(steps):
        x, y = next(batch_iterator)
        x_array.append(x)
        x = Variable(torch.FloatTensor(x), volatile=True)
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
    x_array = np.concatenate(x_array)
    y_preds = torch.cat(y_preds)
    y_gts = torch.cat(y_gts)
    assert y_preds.shape[0] == len(dataset), "All data should be iterated."
    return x_array, y_preds, y_gts, shapes


def get_mc_mt_pred(model, args):
    # setup dataset
    dataset = cornerDataset(args.dataset, args.img_rows,
                            args.img_cols, args.corner, args.buffer, args.sizes)
    data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                             shuffle=False)
    batch_iterator = iter(data_loader)
    shapes = dataset.shapes
    # predict by batch
    x_array, y_preds, y_gts = [], [], []
    steps = len(dataset) // args.batch_size
    if steps * args.batch_size < len(dataset):
        steps += 1
    for step in range(steps):
        x, y = next(batch_iterator)
        x_array.append(x)
        x = Variable(torch.FloatTensor(x), volatile=True)
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
    x_array = np.concatenate(x_array)
    y_preds = torch.cat(y_preds)
    y_gts = torch.cat(y_gts)
    assert y_preds.shape[0] == len(dataset), "All data should be iterated."
    return x_array, y_preds, y_gts, shapes


def pred_to_img(x_array, y_preds, y_gts):
    # convert predict tensor to img & get rgb result
    x_imgs, rgb_results = [], []
    for i in range(y_preds.shape[0]):
        x_img = vision.array_to_img(x_array[i], False)
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
        x_imgs.append(x_img)
        rgb_results.append(rgb_img)
    return x_imgs, rgb_results


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
    if not os.path.exists(os.path.join(Result_DIR, 'corner-comparison')):
        os.makedirs(os.path.join(Result_DIR, 'corner-comparison'))

    compare_imgs = []
    for iter, checkpoint in enumerate(args.checkpoints):
        # Load model
        model = load_checkpoint(checkpoint)
        if args.cuda:
            model.cuda()
        model.eval()

        result_imgs = []
        # slice by number of parts to save memory
        for idx, corner in enumerate(corners):
            # selected part of the data
            args.corner = corner
            print("Predicting by {} at corner {} \r".format(
                checkpoint, corner))

            # get prediction
            if checkpoint.startswith("MC") or checkpoint.startswith("mt"):
                x_array, y_preds, y_gts, shapes = get_mc_mt_pred(model, args)
            else:
                x_array, y_preds, y_gts, shapes = get_pred(model, args)

            # convert tensor to images
            x_imgs, rgb_results = pred_to_img(x_array, y_preds, y_gts)
            del x_array, y_preds, y_gts

            # get original land image
            ori_img = vision.slices_to_img(x_imgs, shapes)
            result_imgs.append(vision.add_barrier(ori_img, args.spaces))
            del ori_img

            # get rgb pred_img
            rgb_img = vision.slices_to_img(rgb_results, shapes)
            result_imgs.append(vision.add_barrier(rgb_img, args.spaces))
            del rgb_img

            gc.collect()
        ori_imgs, rgb_imgs = result_imgs[0::2], result_imgs[1::2]
        compare_imgs += rgb_imgs

    # save final image result
    compare_imgs = ori_imgs + compare_imgs
    final_img = vision.slices_to_img(
        compare_imgs, [len(args.checkpoints) + 1, len(corners)])
    final_img = vision.add_color_bar(final_img, args.spaces[1], 'white')
    name = "{}_{}_buffer_{}_corners.png".format(
        args.target, '_'.join([x.split('_')[0] for x in args.checkpoints]), args.buffer)
    imsave(os.path.join(Result_DIR, 'corner-comparison', name), final_img)
    print("Saving {} ...".format(name))


if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-checkpoints', nargs='+', type=str, default=["FCN_sample.pth", "UNet_sample.pth"],
                        help='checkpoints used for making prediction ')
    parser.add_argument('-target', type=str, default="segmap", choices=['segmap', 'edge'],
                        help='target for model prediction [segmap, edge]')
    parser.add_argument('-edge_fn', type=str, default='canny', choices=['shift', 'canny'],
                        help='method used for edge extraction')
    parser.add_argument('-dataset', type=str, default="RS-2018/test",
                        help='data dir for processing')
    parser.add_argument('-spaces', nargs='+', type=int, default=[20, 30],
                        help='barrier space for merging ')
    parser.add_argument('-color', type=str, default='white', choices=['white', 'black'],
                        help='background color for generated rgb result ')
    parser.add_argument('-sizes', nargs='+', type=int, default=[10, 10],
                        help='number of parts to divided whole area')
    parser.add_argument('-buffer', type=int, default=5,
                        help='buffer area from border ')
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
