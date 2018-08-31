#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @CreateTime:   2018-01-26T16:50:00+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import sys
sys.path.append('./utils')
import os
import numpy as np
import pandas as pd

from torch.utils import data
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import vision
import argparse


Utils_DIR = os.path.dirname(os.path.abspath(__file__))


class NewZealand(data.Dataset):
    """ 'NewZealand' dataset object
    args:
        partition: (str) partition of the data ['nz-train-slc', 'nz-test-slc']
        split: (str) split of the data ['train', 'val', 'all']
    """

    def __init__(self, partition='nz-train-slc', split='train'):
        self.dataset = os.path.join(
            Utils_DIR, '../dataset', partition)

        self._landpath = os.path.join(self.dataset, 'land', '%s')
        self._segpath = os.path.join(self.dataset, 'segmap', '%s')

        # get image ids
        infos = pd.read_csv(os.path.join(
            self.dataset, '{}-infos.csv'.format(split)))
        self.ids = infos['id'].tolist()

        # get label class
        self.nb_class = 1
        # get img sizes
        img_sam = imread(self._landpath % self.ids[0])
        self.img_rows, self.img_cols = img_sam.shape[:2]

    def __len__(self):
        return len(self.ids)


class nzLS(NewZealand):
    """
        return 'Land-Segmap' of NewZealand Dataset
        required data format for normal models \
    """

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_land = imread(self._landpath % img_id)
        img_land = (img_land / 255).astype('float32')

        img_seg = imread(self._segpath % img_id)
        img_seg = (np.expand_dims(img_seg, -1) / 255).astype('float32')

        img_land = img_land.transpose((2, 0, 1))
        img_seg = img_seg.transpose((2, 0, 1))
        return img_land, img_seg

    def show(self, idx):
        img_land = imread(self._landpath % self.ids[idx])
        img_seg = imread(self._segpath % self.ids[idx])

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(11, 5))
        f.suptitle('Sample-{} in NewZealand Dataset'.format(idx))
        ax1.imshow(img_land)
        ax1.set_title('Land Sample')
        ax2.imshow(img_seg, "gray")
        ax2.set_title('Segmap Sample')
        plt.show()

class nzLS8xsub(NewZealand):
    """
        return 'Land-Segmap-Segmap8xsub' of NewZealand Dataset
        required data format for MC-FCN \
        "Automatic Building Segmentation of Aerial Imagery Using Multi-Constraint Fully Convolutional Networks. \
        Remote Sens. 2018"
    """

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_land = imread(self._landpath % img_id)
        img_rows, img_cols = img_land.shape[:2]
        img_land = (img_land / 255).astype('float32')

        img_seg = imread(self._segpath % img_id)
        img_seg_sub = resize(
            img_seg, (img_rows // 8, img_cols // 8), mode='edge')
        # denoise after resing image
        img_seg_sub[img_seg_sub < 0.5] = 0.0
        img_seg_sub[img_seg_sub >= 0.5] = 1.0
        img_seg_sub = (np.expand_dims(img_seg_sub, -1)).astype('float32')

        img_seg = (np.expand_dims(img_seg, -1) / 255).astype('float32')

        img_land = img_land.transpose((2, 0, 1))
        img_seg = img_seg.transpose((2, 0, 1))
        img_seg_sub = img_seg_sub.transpose((2, 0, 1))
        return img_land, img_seg, img_seg_sub

    def show(self, idx):
        img_land = imread(self._landpath % self.ids[idx])
        img_seg = imread(self._segpath % self.ids[idx])
        img_rows, img_cols = img_land.shape[:2]
        img_seg_sub = resize(
            img_seg, (img_rows // 8, img_cols // 8), mode='edge')
        img_seg_sub = (img_seg_sub * 255).astype('uint8')
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
        f.suptitle('Sample-{} in NewZealand Dataset'.format(idx))
        ax1.imshow(img_land)
        ax1.set_title('Land Sample')
        ax2.imshow(img_seg, 'gray')
        ax2.set_title('Segmap Sample')
        ax3.imshow(img_seg_sub, 'gray')
        ax3.set_title('Segmap8xsub Sample')
        plt.show()


class nzLSE(NewZealand):
    """
        return 'Land-Segmap-Edge' of NewZealand Dataset
        required data format for BR-Net \
        "A Boundary Regulated Network for Accurate Roof Segmentation and Outline Extraction \
        Remote Sens. 2018."
    """

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_land = imread(self._landpath % img_id)
        img_land = (img_land / 255).astype('float32')

        img_seg = imread(self._segpath % img_id)
        img_edge = vision.shift_edge(img_seg)

        # img_seg = rgb2gray(img_seg)
        # img_edge = np.expand_dims(vision.canny_edge(img_seg), dim=-1)
        # img_edge = (img_edge / 255).astype("float32")

        img_seg = (np.expand_dims(img_seg, -1) / 255).astype('float32')
        img_edge = (np.expand_dims(img_edge, -1) / 255).astype('float32')

        img_land = img_land.transpose((2, 0, 1))
        img_seg = img_seg.transpose((2, 0, 1))
        img_edge = img_edge.transpose((2, 0, 1))

        return img_land, img_seg, img_edge

    def show(self, idx):
        img_land = imread(self._landpath % self.ids[idx])
        img_seg = imread(self._segpath % self.ids[idx])
        img_edge = vision.shift_edge(img_seg)

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
        f.suptitle('Sample-{} in NewZealand Dataset'.format(idx))
        ax1.imshow(img_land)
        ax1.set_title('Land Sample')
        ax2.imshow(img_seg, 'gray')
        ax2.set_title('Segmap Sample')
        ax3.imshow(img_edge, 'gray')
        ax3.set_title('Edge Sample')
        plt.show()

if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-partition', type=str, default='nz-train-slc',
                        help='partition within of the dataset ')
    parser.add_argument('-split', type=str, default='all',
                        help='split of the data within ["train","val","test","all"]')
    args = parser.parse_args()

    # NewZealand dataset
    lsdata = nzLS(args.partition, args.split)
    land, seg = lsdata[0]
    lsdata.show(0)

    ls8xdata = nzLS8xsub(args.partition, args.split)
    land, seg, seg8x = ls8xdata[0]
    ls8xdata.show(0)

    lsedata = nzLSE(args.partition, args.split)
    land, seg, edge = lsedata[0]
    lsedata.show(0)
