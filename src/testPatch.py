#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""

import os
import time
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
        if 'IRRG' in root:
            self.data = os.path.join(Data_DIR, root.replace('IRRG',''))
            self._srcpath = os.path.join(self.data, 'Ortho_IRRG', '%s')
            self._tarpath = os.path.join(self.data, 'Mask', '%s')
            textfile = 'IRRG-{}.txt'.format(split)
        elif 'RGB' in root:
            self.data = os.path.join(Data_DIR, root.replace('RGB',''))
            self._srcpath = os.path.join(self.data, 'Ortho_RGB', '%s')
            self._tarpath = os.path.join(self.data, 'Mask', '%s')
            textfile = 'RGB-{}.txt'.format(split)
        elif 'RGBIR' in root:
            self.data = os.path.join(Data_DIR, root.replace('RGBIR',''))
            self._srcpath = os.path.join(self.data, 'Ortho_RGBIR', '%s')
            self._tarpath = os.path.join(self.data, 'Mask', '%s')
            textfile = 'RGBIR-{}.txt'.format(split)
        else:
            self.data = os.path.join(Data_DIR, root)
            self._srcpath = os.path.join(self.data, 'Ortho', '%s')
            self._tarpath = os.path.join(self.data, 'Mask', '%s')
            textfile = '{}.txt'.format(split)
        # get testing files
        with open(os.path.join(self.data, textfile), 'r') as f:
            self.files = [line.strip() for line in f.readlines()]
        # get reference
        self.ref = pd.read_csv(os.path.join(self.data, 'ref.csv'))
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
        tar_img = imread(self._tarpath % self.files[idx])
        assert src_img.shape[:2] == tar_img.shape[:2], "Image dimension must be consistent."
        if len(tar_img.shape) == 3:
            tar_img = self.color2label(tar_img, self.ref)
        else:
            tar_img = (tar_img / 255).astype('uint8')
        # extract slices
        x, shapes = vision.img_to_slices(src_img, self.img_rows, self.img_cols)
        y, shapes = vision.img_to_slices(tar_img, self.img_rows, self.img_cols)
        # convert to tensor
        x = vision.xslices_to_tensor(x)
        y = vision.yslices_to_tensor(y, self.cmap)
        return x, y, shapes


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
    evaluators = [metrics.OAAcc(), metrics.Precision(), metrics.Recall(), 
                  metrics.F1Score(), metrics.Kappa(), metrics.Jaccard()]
    # prediction
    for checkpoint in args.checkpoints:
        model, is_multi = load_checkpoint(checkpoint, args.cuda)
        performs = [[] for i in range(len(evaluators))]
        for idx in range(len(data)):
            print("Handling {} by {} \r".format(data.files[idx], checkpoint))
            x, y, shapes = data.slice_by_id(idx)
            # generate prediction
            with torch.set_grad_enabled(False):
                for step in range(0, x.shape[0], args.batch_size):
                    x_batch = x[step:step+args.batch_size]
                    y_batch = y[step:step+args.batch_size]
                    if args.cuda:
                        x_batch = x_batch.cuda()
                        y_batch = y_batch.cuda()
                    if is_multi:
                        y_pred = model(x_batch)[0].detach()
                    else:
                        y_pred = model(x_batch).detach()
                    # get performance
                    for i, evaluator in enumerate(evaluators):
                        performs[i].append(evaluator(y_pred, y_batch)[0].item())

        performs = [(sum(p) / len(p)) for p in performs]
        performs = pd.DataFrame([[time.strftime("%h_%d"), checkpoint] + performs],
                               columns=['time', 'checkpoint'] + [repr(x) for x in evaluators])
        # save performance
        log_path = os.path.join(Result_DIR, "patchPerforms.csv")
        if os.path.exists(log_path):
            perform = pd.read_csv(log_path)
        else:
            perform = pd.DataFrame([])
        perform = perform.append(performs, ignore_index=True)
        perform.to_csv(log_path, index=False, float_format="%.3f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-checkpoints', nargs='+', type=str, default=["FCN8s-3*6*24-Vaihingen_iter_1000.pth"],
                        help='checkpoints used for making prediction ')
    parser.add_argument('-root', type=str, default="Vaihingen",
                        help='root dir of test data ')
    parser.add_argument('-img_rows', type=int, default=224,
                        help='img rows for croping ')
    parser.add_argument('-img_cols', type=int, default=224,
                        help='img cols for croping ')
    parser.add_argument('-has_anno', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='has/hasn\'t annotation data')
    parser.add_argument('-batch_size', type=int, default=24,
                        help='batch size for model prediction ')
    parser.add_argument('-cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')
    args = parser.parse_args()
    main(args)
