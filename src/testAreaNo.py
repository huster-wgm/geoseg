#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""

import os
import time
import cv2
import torch
import vision
import metrics
import argparse
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)

DIR = os.path.dirname(os.path.abspath(__file__))
Data_DIR = os.path.join(DIR, '../data')
Result_DIR = os.path.join(DIR, '../result')
Checkpoint_DIR = os.path.join(DIR, '../checkpoint')



class tileData(object):
    """
    Testing tile data
    args:
        root: (str) root dir of data for evalutation, e.g. Vaihingen
        split: (str) split of the data ['train', 'val', 'all']
    """

    def __init__(self, root='Vaihingen', img_rows=224, img_cols=224, split='test'):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self._srcpath = os.path.join(Data_DIR, root, 'Ortho', '%s')
        self._tarpath = os.path.join(Data_DIR, root, 'Mask', '%s')
        # get testing files
        with open(os.path.join(Data_DIR, root, '{}.txt'.format(split)), 'r') as f:
            self.files = [line.strip() for line in f.readlines()]
        # get reference
        self.ref = pd.read_csv(os.path.join(Data_DIR, root, 'ref.csv'))
        self.cmap = (self.ref.iloc[:,1:]).values

    def __len__(self):
        return len(self.files)

    def color2label(self, img, ref):
        row, col, ch = img.shape
        tmp = np.zeros((row, col), 'uint8')
        for idx, row in ref.iterrows():
            tmp[np.logical_and(
                np.logical_and(img[:, :, 0] == row[1], img[:, :, 1] == row[2]),
                img[:, :, 2] == row[3])] = idx
        return tmp

    def slice_by_id(self, idx):
        src_img = imread(self._srcpath % self.files[idx])
        src_img = cv2.resize(src_img, None, fx=3.3, fy=3.3)
        rows, cols = src_img.shape[:2]
        src_img = src_img[:rows//2, :cols//2, :]
        print("SOURCE IMAGE:", src_img.shape)
        # extract slices
        x, shapes = vision.img_to_slices(src_img, self.img_rows, self.img_cols)
        # convert to tensor
        x = vision.xslices_to_tensor(x)
        return x, shapes


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
    # init
    data = tileData(args.root, args.img_rows, args.img_cols)
    # prediction
    for checkpoint in args.checkpoints:
        model, is_multi = load_checkpoint(checkpoint, args.cuda)
        Save_DIR = os.path.join(Result_DIR, "area", checkpoint.split("_")[0])
        if not os.path.exists(Save_DIR):
            os.makedirs(Save_DIR)
        for idx in range(len(data)):
            print("Handling {} by {} \r".format(data.files[idx], checkpoint))
            x, shapes = data.slice_by_id(idx)
            # get prediction
            y_preds =[]
            with torch.set_grad_enabled(False):
                for step in range(0, x.shape[0], args.batch_size):
                    x_batch = x[step:step+args.batch_size]
                    if args.cuda:
                        x_batch = x_batch.cuda()
                    # generate prediction
                    y_pred = model(x_batch)
                    if is_multi:
                        y_pred = y_pred[0]
                    if args.cuda:
                        y_preds.append(y_pred.cpu()) 
                        
            y_preds = torch.cat(y_preds, 0)
            assert y_preds.shape[0] == x.shape[0], "All data should be iterated."
            del x
            pred_img = vision.slices_to_img(
                vision.ytensor_to_slices(y_preds, data.cmap), shapes)
            # merge slices into image & save result image
            imsave(os.path.join(Save_DIR, data.files[idx]), pred_img, compress=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-checkpoints', nargs='+', type=str, default=["UNet-3*6*24-PotsdamRGB_iter_100000.pth"],
                        help='checkpoints used for making prediction ')
    parser.add_argument('-root', type=str, default="Xiaoya",
                        help='root dir of test data ')
    parser.add_argument('-img_rows', type=int, default=224,
                        help='img rows for croping ')
    parser.add_argument('-img_cols', type=int, default=224,
                        help='img cols for croping ')
    parser.add_argument('-batch_size', type=int, default=24,
                        help='batch size for model prediction ')
    parser.add_argument('-cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()
    main(args)
