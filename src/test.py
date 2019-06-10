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
import torch.optim as optim
from skimage.io import imread, imsave


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)

Src_DIR = os.path.dirname(os.path.abspath(__file__))
Data_DIR = os.path.join(Src_DIR, '../data')
Result_DIR = os.path.join(Src_DIR, '../result')
Checkpoint_DIR = os.path.join(Src_DIR, '../checkpoint')

refs = np.array([
    ["Impervious_surfaces", 255, 255, 255],
    ["Building", 0, 0, 255],
    ["Low vegetation", 0, 255, 255],
    ["Tree", 0, 255, 0],
    ["Car", 255, 255, 0],
    ["Background", 255, 0, 0],
])


class areaData(object):
    """
    ISPRS benchmark dataset Object
    args:
        dataset: (str) data for evalutation, e.g. Vaihingen
        split: (str) split of the data ['train', 'val', 'all']
    """

    def __init__(self, dataset='Vaihingen', img_rows=224, img_cols=224, split='test'):
        self.img_rows = img_rows
        self.img_cols = img_cols
        if 'IRRG' in dataset:
            self.data = os.path.join(Data_DIR, dataset.replace('IRRG',''))
            self._srcpath = os.path.join(self.data, 'Ortho_IRRG', '%s')
            self._tarpath = os.path.join(self.data, 'Mask', '%s')
            textfile = 'IRRG-{}.txt'.format(split)
        elif 'RGB' in dataset:
            self.data = os.path.join(Data_DIR, dataset.replace('RGB',''))
            self._srcpath = os.path.join(self.data, 'Ortho_RGB', '%s')
            self._tarpath = os.path.join(self.data, 'Mask', '%s')
            textfile = 'RGB-{}.txt'.format(split)
        elif 'RGBIR' in dataset:
            self.data = os.path.join(Data_DIR, dataset.replace('RGBIR',''))
            self._srcpath = os.path.join(self.data, 'Ortho_RGBIR', '%s')
            self._tarpath = os.path.join(self.data, 'Mask', '%s')
            textfile = 'RGBIR-{}.txt'.format(split)
        else:
            self.data = os.path.join(Data_DIR, dataset)
            self._srcpath = os.path.join(self.data, 'Ortho', '%s')
            self._tarpath = os.path.join(self.data, 'Mask', '%s')
            textfile = '{}.txt'.format(split)
        # get image ids
        self.ids = []
        with open(os.path.join(self.data, textfile), 'r') as f:
            for line in f.readlines():
                self.ids.append(line.strip())

        # get label reference
        self.refs = refs
        self.nb_class = self.refs.shape[0]

    def __len__(self):
        return len(self.ids)

    def slice_by_id(self, idx):
        self.src_img = imread(self._srcpath % self.ids[idx])
        self.tar_img = imread(self._tarpath % self.ids[idx])
        assert self.src_img.shape[:2] == self.tar_img.shape[:
                                                            2], "Image dimension must be consistent."
        # extract slices from source and target imagery
        X_slices, shapes = vision.img_to_slices(self.src_img, self.img_rows, self.img_cols)
        y_slices, shapes = vision.img_to_slices(self.tar_img, self.img_rows, self.img_cols)
        self.shapes = shapes
        self.X = vision.slices_to_tensor(X_slices)
        self.y = vision.slices_to_tensor(y_slices, self.refs)


def load_checkpoint(checkpoint, cuda):
    from models.fcn import FCN8s, FCN16s, FCN32s
    from models.unet import UNet
    from models.segnet import SegNet
    from models.resunet import ResUNet
    from models.fpn import FPN
    from models.mcfcn import MCFCN
    from models.brnet import BRNetv0, BRNetv1, BRNetv2, BRNetv3, BRNetv4
    assert os.path.exists("{}/{}".format(Checkpoint_DIR, checkpoint)
                          ), "{} does not exists.".format(checkpoint)
    method = checkpoint.split('_')[0]
    model = method.split('-')[0]
    src_ch, tar_ch, base_kernel = [int(x) for x in method.split('-')[1].split("*")]
    net = eval(model)(src_ch, tar_ch, base_kernel)
    net.load_state_dict(
        torch.load(os.path.join(Checkpoint_DIR, checkpoint)))
    if cuda:
        net.cuda()
    print("Loaded checkpoint: {}".format(checkpoint))
    return net.eval()


def get_pred(model, X, args):
    # predict by batch
    y_preds =[]
    for step in range(0, X.shape[0], args.batch_size):
        x = X[step:step+args.batch_size]
        if args.cuda:
            x = x.cuda()
        # generate prediction
        y_pred = model(x)
        y_preds.append(y_pred.data)

    y_preds = torch.cat(y_preds, 0)
    if args.cuda:
        y_preds = y_preds.cpu()
    assert y_preds.shape[0] == X.shape[0], "All data should be iterated."
    return y_preds



def main(args):
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("GPUs are not available, please run at cpu mode")
    # init
    data = areaData(args.dataset, args.img_rows, args.img_cols)
    evaluators = [metrics.OAAcc(), metrics.Precision(), metrics.Recall(), 
                  metrics.F1Score(), metrics.Kappa(), metrics.Jaccard()]
    # prediction
    for checkpoint in args.checkpoints:
        model = load_checkpoint(checkpoint, args.cuda)
        Save_DIR = os.path.join(Result_DIR, checkpoint.split("_")[0])
        if not os.path.exists(Save_DIR):
            os.makedirs(Save_DIR)
        performs = [0 for i in range(len(evaluators))]
        for idx in range(len(data)):
            print("Predicting by {} of {} \r".format(checkpoint, data.ids[idx]))
            data.slice_by_id(idx)
            # get prediction
            y_preds = get_pred(model, data.X, args)
            pred_img = vision.slices_to_img(vision.tensor_to_slices(y_preds, data.refs), data.shapes)
            y_img = vision.slices_to_img(vision.tensor_to_slices(data.y, data.refs), data.shapes)
            # merge slices into image & save result image
            imsave(os.path.join(Save_DIR, data.ids[idx]),
                   pred_img,  compress=6)
            # img2input
            pred_img_tensor = torch.FloatTensor(
                np.expand_dims(vision.img_to_label(pred_img, data.refs).transpose((2,0,1)), axis=0))
            y_img_tensor = torch.FloatTensor(
                np.expand_dims(vision.img_to_label(y_img, data.refs).transpose((2,0,1)), axis=0))
            # get performance
            for idx, evaluator in enumerate(evaluators):
                performs[idx] += evaluator(pred_img_tensor, y_img_tensor)[0]

        performs = ["%0.3f" % float(value / len(data)) for value in performs]

        performs = pd.DataFrame([[time.strftime("%h%d_%H"), checkpoint.split('.pth')[0]] + performs],
                               columns=['time', 'checkpoint'] + [repr(x) for x in evaluators])

        # save performance
        log_path = os.path.join(Result_DIR, "performs.csv")
        if os.path.exists(log_path):
            perform = pd.read_csv(log_path)
        else:
            perform = pd.DataFrame([])
        perform = perform.append(performs, ignore_index=True)
        perform.to_csv(log_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-checkpoints', nargs='+', type=str, default=["FCN8s-3*6*24-Vaihingen_iter_1000.pth"],
                        help='checkpoints used for making prediction ')
    parser.add_argument('-dataset', type=str, default="Vaihingen",
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
