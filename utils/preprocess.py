#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @CreateTime:   2018-01-26T16:50:00+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import sys
sys.path.append('./utils')
import glob
import json
import shutil
import random
import itertools
import numpy as np
import pandas as pd

from skimage.io import imread, imsave
from skimage.transform import resize
import argparse


Utils_DIR = os.path.dirname(os.path.abspath(__file__))

class Processor(object):
    def __init__(self):
        print("Baic processor")

    def save_infos(self, df):
        all_file = os.path.join(self.save_dir, 'all-infos.csv')
        df.to_csv(all_file, index=False)
        nb_list = list(range(df.shape[0]))
        tv_edge = int(df.shape[0] * self.split[0])
        vt_edge = int(df.shape[0] * (1 - self.split[2]))
        # shuffle list
        random.shuffle(nb_list)
        train_df = df.iloc[nb_list[:tv_edge], :]
        train_df.to_csv(os.path.join(self.save_dir, 'train-infos.csv'), index=False)
        val_df = df.iloc[nb_list[tv_edge:vt_edge], :]
        val_df.to_csv(os.path.join(self.save_dir, 'val-infos.csv'), index=False)
        test_df = df.iloc[nb_list[vt_edge:], :]
        test_df.to_csv(os.path.join(self.save_dir, 'test-infos.csv'), index=False)

    def save_slices(self, img_slices, folder):
        os.mkdir(os.path.join(self.save_dir, folder))
        for i in range(len(img_slices)):
            imsave(os.path.join(self.save_dir, folder, "img_{0}.png".format(i)),
                   img_slices[i])
        return 0

class singleProcessor(Processor):
    def __init__(self, data, img_rows, img_cols,
                 stride=None,
                 threshold=0.1,
                 edge_buffer=0.1,):
        self.src_path = os.path.join(Utils_DIR, '../data', data, "source.tif")
        self.tar_path = os.path.join(Utils_DIR, '../data', data, "target.tif")
        self.data = data

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.threshold = threshold
        self.edge_buffer = edge_buffer
        self.split = [0.6, 0.2, 0.2]
        self.stride = stride if stride else img_rows

        print("Aligning source and target image ...")
        self._img_align()

    def _read_tfw(self, tfw):
        with open(tfw) as f:
            params = []
            for line in f.readlines():
                param = float(line.strip())
                params.append(param)
            return params

    def _img_align(self):
        self.src_img = imread(self.src_path)
        self.tar_img = imread(self.tar_path)
        # extend tar image dimension
        # read coordinate params info from tfw
        assert os.path.exists(self.src_path.replace(
            '.tif', '.tfw')), "Source tfw doesn't exist, try slicing mode instead."
        assert os.path.exists(self.tar_path.replace(
            '.tif', '.tfw')), "Target tfw doesn't exist, try slicing mode instead."
        assert os.path.exists(self.tar_path.replace(
            '.tif', '.geojson')), "Target geojson doesn't exist, try slicing mode instead."
        tar_params = self._read_tfw(self.tar_path.replace('.tif', '.tfw'))
        src_params = self._read_tfw(self.src_path.replace('.tif', '.tfw'))
        with open(self.tar_path.replace('.tif', '.geojson'), 'r') as f:
            topos = json.load(f)['features']
        self.topos = topos
        assert len(src_params) == len(
            tar_params), "Number of params should be equal."
        assert src_params[:4] == tar_params[:4], "Resolution should be the same."

        # pixel scale along with x and y axis
        self.x_axis_scale = src_params[0]
        self.y_axis_scale = src_params[3]
        # x, y coordinate for src and tar
        x_1, y_1 = src_params[4:]
        x_2, y_2 = tar_params[4:]
        # initialize bounds[minx, miny, maxx, maxy] of consistent area
        bounds = [0, 0, 0, 0]
        # coumpute shifting pixels
        x_pixel = int((x_2 - x_1) / self.x_axis_scale)
        y_pixel = int((y_2 - y_1) / self.y_axis_scale)

        # alignment by original point (0, 0)
        # align by x-axis
        if x_pixel >= 0:
            # chop src image in x-axis -> cols
            self.src_img = self.src_img[:, x_pixel:]
            bounds[0] = x_2
        else:
            # chop ouline image in x-axis -> cols
            self.tar_img = self.tar_img[:, abs(x_pixel):]
            bounds[0] = x_1
        # align by y-axis
        if y_pixel >= 0:
            # chop src image in y-axis -> rows
            self.src_img = self.src_img[y_pixel:, :]
            bounds[3] = y_2
        else:
            # chop tar image in y-axis -> rows
            self.tar_img = self.tar_img[abs(y_pixel):, :]
            bounds[3] = y_1

        # crop max consistent area
        rows_l, cols_l = self.src_img.shape[:2]
        rows_o, cols_o = self.tar_img.shape[:2]
        # crop by rows
        self.tar_img = self.tar_img[:min(rows_l, rows_o), :]
        self.src_img = self.src_img[:min(rows_l, rows_o), :]
        # crop by cols
        self.tar_img = self.tar_img[:, :min(cols_l, cols_o)]
        self.src_img = self.src_img[:, :min(cols_l, cols_o)]
        bounds[2] = self.x_axis_scale * min(cols_l, cols_o) + bounds[0]
        bounds[1] = self.y_axis_scale * min(rows_l, rows_o) + bounds[3]
        self.bounds = bounds

    def _get_bounds(self, polygon):
        # return bounds(minx, miny, maxx, maxy) of the polygon
        bounds = [0, 0, 0, 0]
        poly = np.array(polygon['geometry']['coordinates'])[0]
        try:
            bounds[:2] = poly.min(axis=0)
            bounds[2:] = poly.max(axis=0)
            return bounds
        except:
            # print("wrong polygon")
            return None

    def extract_by_slice(self):
        # make save dirs
        self.save_dir = os.path.join(
            Utils_DIR, '../dataset', self.data + '-slc')
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

        print("Processing via slicing window")
        assert self.src_img.shape[:2] == self.tar_img.shape[:
                                                            2], "Image dimension must be consistent."
        # extract slices from source and target imagery
        rows, cols = self.src_img.shape[:2]
        row_range = range(0, rows - self.img_rows, self.stride)
        col_range = range(0, cols - self.img_cols, self.stride)
        print("\t Original: img_rows : {}; img_cols : {}".format(rows, cols))
        print("\t Original: nb_rows : {}; nb_cols : {}".format(
            len(row_range), len(col_range)))
        X_slices, y_slices, posi_rates = [], [], []
        for i, j in itertools.product(row_range, col_range):
            img_src = self.src_img[i:i + self.img_rows,
                                   j:j + self.img_cols]
            img_tar = self.tar_img[i:i + self.img_rows,
                                   j:j + self.img_cols]
            posi_rate = round(np.sum(img_tar == 255) /
                              (self.img_rows * self.img_cols), 3)
            if posi_rate >= self.threshold:
                X_slices.append(img_src)
                y_slices.append(img_tar)
                posi_rates.append(posi_rate)

        if self.threshold == 0:
            assert len(y_slices) == len(row_range) * len(
                col_range), "If threshold==0, save image number should be consistent with original size."
        _statistic = [len(y_slices), len(row_range), len(col_range),
                      self.img_rows, self.img_cols, np.mean(posi_rates)]
        _file = os.path.join(self.save_dir, 'statistic.csv')
        pd.DataFrame([_statistic],
                     columns=["nb-samples", "nb_rows", "nb_cols", "img_rows", "img_cols", "mean-posi"]).to_csv(_file, index=False)

        # save infos
        infos = pd.DataFrame(columns=['id', 'posi_rate'])
        infos['id'] = ['img_{}.png'.format(i) for i in range(len(y_slices))]
        infos['posi_rate'] = posi_rates
        self.save_infos(infos)

        # save slices
        self.save_slices(X_slices, "land")
        self.save_slices(y_slices, "segmap")

    def extract_by_vector(self):
        # make save dirs
        self.save_dir = os.path.join(
            Utils_DIR, '../dataset', self.data + '-vec')
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

        print("Processing via vector")
        X_slices, y_slices = [], []
        posi_rates, ori_sizes = [], []
        # count number of polygon out of range
        counts = 0
        for topo in self.topos:
            # bounds = (minx, miny, maxx, maxy)
            try:
                bounds = self._get_bounds(topo)
                cen_pixel_x = int(
                    ((bounds[2] + bounds[0]) / 2 - self.bounds[0]) / self.x_axis_scale)
                cen_pixel_y = int(
                    ((bounds[3] + bounds[1]) / 2 - self.bounds[3]) / self.y_axis_scale)
                max_width = (bounds[2] - bounds[0]) / self.x_axis_scale
                max_height = (bounds[1] - bounds[3]) / self.y_axis_scale
                pixels = int((self.edge_buffer + 1) *
                             max(max_width, max_height) / 2)

                # crop building from src and tar with buffer
                extract_src = self.src_img[cen_pixel_y - pixels:cen_pixel_y + pixels,
                                           cen_pixel_x - pixels:cen_pixel_x + pixels]
                extract_tar = self.tar_img[cen_pixel_y - pixels:cen_pixel_y + pixels,
                                           cen_pixel_x - pixels:cen_pixel_x + pixels]
                extract_src = resize(
                    extract_src, (self.img_rows, self.img_cols), mode='edge')
                extract_tar = resize(
                    extract_tar, (self.img_rows, self.img_cols), mode='edge')

                extract_src = (extract_src * 255).astype("uint8")
                extract_tar = (extract_tar * 255).astype("uint8")
                # denoise after resing image
                extract_tar[extract_tar < 128] = 0
                extract_tar[extract_tar >= 128] = 255
                posi_rate = round(np.sum(extract_tar == 255) /
                                  (self.img_rows * self.img_cols), 3)
                X_slices.append(extract_src)
                y_slices.append(extract_tar)
                posi_rates.append(posi_rate)
                ori_sizes.append(2 * pixels)
            except:
                counts += 1
                print("\t Skip polygon which is out of range")

        print("\t Totally %d of ploygons are out of scope." % counts)

        _statistic = [len(y_slices), self.img_rows,
                      self.img_cols, np.mean(posi_rates)]
        _file = os.path.join(self.save_dir, 'statistic.csv')
        pd.DataFrame([_statistic],
                     columns=["nb-samples", "img_rows", "img_cols", "mean-posi"]).to_csv(_file, index=False)

        # save infos
        infos = pd.DataFrame(columns=['id', 'ori_size', 'posi_rate'])
        infos['id'] = ['img_{}.png'.format(i) for i in range(len(y_slices))]
        infos['ori_size'] = ori_sizes
        infos['posi_rate'] = posi_rates
        self.save_infos(infos)

        # save slices
        self.save_slices(X_slices, "land")
        self.save_slices(y_slices, "segmap")


class multiProcessor(Processor):
    """Image Data for preprocessing multi-label image
    args:
        data: (str) root of the dataset e.g. 'Vaihingen'
        split: (float) split of train-val distribution
    """

    def __init__(self, data, img_rows, img_cols,
                 stride=None,):
        self.data = os.path.join(Utils_DIR, '../data', data)

        # make save dirs
        self.save_dir = os.path.join(Utils_DIR, '../dataset', data + "-slc")
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.split = [0.6, 0.2, 0.2]

        self._srcpath = os.path.join(self.data, 'TOP', '%s')
        self._tarpath = os.path.join(self.data, 'GT', '%s')

        # get image ids
        self.ids = []
        with open(os.path.join(self.data, 'train.txt'), 'r') as f:
            for line in f.readlines():
                self.ids.append(line.strip())
        self.stride = stride if stride else img_rows

    def extract_by_slice(self):

        X_slices, y_slices = [], []
        ori_file, ori_size, ori_pos = [], [], []
        for img_id in self.ids:
            self.src_img = imread(self._srcpath % img_id)
            self.tar_img = imread(self._tarpath % img_id)
            assert self.src_img.shape[:2] == self.tar_img.shape[:
                                                                2], "Image dimension must be consistent."
            # extract slices from source and target imagery
            rows, cols = self.src_img.shape[:2]
            row_range = range(0, rows - self.img_rows, self.stride)
            col_range = range(0, cols - self.img_cols, self.stride)
            print("Processing {}...".format(img_id))
            print("\t Original: img_rows : {}; img_cols : {}".format(rows, cols))
            print("\t Original: nb_rows : {}; nb_cols : {}".format(
                len(row_range), len(col_range)))

            for i, j in itertools.product(row_range, col_range):
                img_src = self.src_img[i:i + self.img_rows,
                                       j:j + self.img_cols]
                img_tar = self.tar_img[i:i + self.img_rows,
                                       j:j + self.img_cols]
                X_slices.append(img_src)
                y_slices.append(img_tar)
                ori_file.append(img_id)
                ori_size.append("{},{}".format(
                    len(row_range), len(col_range)))
                ori_pos.append("{},{}".format(i, j))

        # save infos
        infos = pd.DataFrame(columns=['id', 'tile', 'ori_size', 'ori_pos'])
        infos['id'] = ['img_{}.png'.format(i) for i in range(len(y_slices))]
        infos['tile'] = ori_file
        infos['ori_size'] = ori_size
        infos['ori_pos'] = ori_pos
        self.save_infos(infos)

        # save slices
        self.save_slices(X_slices, "land")
        self.save_slices(y_slices, "segmap")


if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-data', type=str, default="RS-2018-compress",
                        help='data dir for processing')
    parser.add_argument('-is_multi', type=bool, default=False,
                        help='where to use multi-bands processors')
    parser.add_argument('-mode', type=str, default="slice",
                        choices=['slice', 'vector', 'both'],
                        help='croping mode ')
    parser.add_argument('-img_rows', type=int, default=224,
                        help='img rows for croping ')
    parser.add_argument('-img_cols', type=int, default=224,
                        help='img cols for croping ')
    parser.add_argument('-stride', type=int, default=224,
                        help='img cols for croping ')
    parser.add_argument('-threshold', type=float, default=0.1,
                        help='hourse cover rate to eliminate')
    parser.add_argument('-edge_buffer', type=float, default=0.1,
                        help='buffer area from edge')
    args = parser.parse_args()
    if args.is_multi:
        processor = multiProcessor(
            args.data, args.img_rows, args.img_cols, args.stride)
        processor.extract_by_slice()
    else:
        processor = singleProcessor(args.data, args.img_rows, args.img_cols,
                                    args.stride, args.threshold, args.edge_buffer)
        if args.mode == 'slice':
            processor.extract_by_slice()
        elif args.mode == 'vector':
            processor.extract_by_vector()
        else:
            processor.extract_by_slice()
            processor.extract_by_vector()
