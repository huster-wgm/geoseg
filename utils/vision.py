#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @CreateTime:   2018-01-26T16:50:00+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import re
import os
import itertools
import numpy as np
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import erosion, dilation, disk
from skimage import feature
from skimage.color import rgb2gray


def cls_to_label(img_cls, nb_class):
    """converting class to sparse label
    args:
        img_cls: (2d array) img class
        nb_class: total number of class
    return img_label
    """
    rows, cols = img_cls.shape[: 2]
    img_label = np.zeros((rows, cols, nb_class), "float32")
    for cls in range(nb_class):
        img_label[img_cls == cls, cls] = 1.0
    return img_label


def img_to_cls(img, refs):
    """converting RGB grand turth to labels class
    args:
        img: (3d array) RGB image
        refs: (2d array ) reference  ['label', 'R-value', 'G-value', 'B-value']
    return img_cls
    """
    rows, cols = img.shape[: 2]
    nb_class = refs.shape[0]
    img_cls = np.zeros((rows, cols), "uint8")
    for cls in range(nb_class):
        img_tmp = np.zeros((rows, cols, 3), "uint8")
        img_tmp[:, :, 0] = refs[cls, 1]
        img_tmp[:, :, 1] = refs[cls, 2]
        img_tmp[:, :, 2] = refs[cls, 3]
        img_consit = np.sum(img_tmp == img, axis=-1)
        img_cls[img_consit == 3] = cls
    return img_cls


def img_by_layer(img, refs):
    """converting RGB grand turth to labels class
    args:
        img: (3d array) RGB image
        refs: (2d array ) reference  ['label', 'R-value', 'G-value', 'B-value']
    return imgs
    """
    rows, cols = img.shape[: 2]
    nb_class = refs.shape[0]
    img_layers = []
    for cls in range(nb_class):
        layer = np.zeros((rows, cols, 4), "uint8")
        img_tmp = np.zeros((rows, cols, 3), "uint8")
        img_tmp[:, :, 0] = refs[cls, 1]
        img_tmp[:, :, 1] = refs[cls, 2]
        img_tmp[:, :, 2] = refs[cls, 3]
        img_consit = np.sum(img_tmp == img, axis=-1)
        layer[img_consit == 3, :3] = refs[cls, 1:]
        layer[img_consit == 3, -1] = 255
        img_layers.append(layer)
    return img_layers


def img_to_label(img, refs):
    """converting RGB grand turth to labels
    args:
        img: (3d array) RGB image
        refs: (2d array ) reference  ['label', 'R-value', 'G-value', 'B-value']
    return img_label
    """
    img_cls = img_to_cls(img, refs)
    img_label = cls_to_label(img_cls, refs.shape[0])
    return img_label


def label_to_cls(img_label):
    """converting sparse label to img cls
    args:
        img_label: (3d array) img label
    return img_cls
    """
    img_cls = np.argmax(img_label, axis=-1)
    return img_cls


def cls_to_img(img_cls, refs):
    """converting img class to RGB img
    args:
        img_cls: (2d array) img class
        refs: (ndarray ) reference  ['label', 'R-value', 'G-value', 'B-value']
    return img_cls
    """
    rows, cols = img_cls.shape[: 2]
    nb_class = refs.shape[0]
    img = np.zeros((rows, cols, 3), np.uint8)
    for cls in range(nb_class):
        cls_rgb = refs[cls, 1:]
        img[img_cls == cls] = cls_rgb
    return img


def label_to_img(img_label, refs):
    """converting img labels to RGB img
    args:
        img_label: (3d array) img label
        refs: (ndarray ) reference  ['label', 'R-value', 'G-value', 'B-value']
    return img
    """
    img_cls = label_to_cls(img_label)
    img = cls_to_img(img_cls, refs)
    return img


def img_to_tensor(img):
    """
    Convert img to img tensor
    args:
      img: (ndarray) unit8
    return img_tensor
    """
    img = (img / 255).astype('float32')
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img = img.transpose((2, 1, 0))
    img = np.expand_dims(img, axis=0)
    return torch.FloatTensor(img)


def get_idx_by_range(img_shapes, row_range, col_range):
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
    assert row_range[1] <= img_shapes[0], "max range of row should less that nb_rows"
    assert col_range[1] <= img_shapes[1], "max range of col should less that nb_cols"
    selected_idx = []
    row_rg = range(row_range[0], row_range[1], 1)
    col_rg = range(col_range[0], col_range[1], 1)
    for row, col in itertools.product(row_rg, col_rg):
        selected_idx.append(row * img_shapes[1] + col)

    selected_shapes = [row_range[1] -
                       row_range[0], col_range[1] - col_range[0]]
    return selected_idx, selected_shapes


def img_to_slices(img, img_rows, img_cols):
    """
    Convert image into slices
        args:
            .img: input image
            .img_rows: img_rows of slice
            .img_cols: img_rows of slice
        return slices, shapes[nb_rows, nb_cols]
    """
    nb_rows = img.shape[0] // img_rows
    nb_cols = img.shape[1] // img_cols
    slices = []
    # generate img slices
    for i, j in itertools.product(range(nb_rows), range(nb_cols)):
        slice = img[i * img_rows: i * img_rows + img_rows,
                    j * img_cols:j * img_cols + img_cols]
        slices.append(slice)
    return slices, [nb_rows, nb_cols]


def slices_to_img(slices, shapes):
    """
    Restore slice into image
        args:
            slices: image slices
            shapes: [nb_rows, nb_cols] of original image
        return img
    """
    # set img placeholder
    if len(slices[0].shape) == 3:
        img_rows, img_cols, in_ch = slices[0].shape
        img = np.zeros(
            (img_rows * shapes[0], img_cols * shapes[1], in_ch), np.uint8)
    else:
        img_rows, img_cols = slices[0].shape
        img = np.zeros((img_rows * shapes[0], img_cols * shapes[1]), np.uint8)
    # merge
    for i, j in itertools.product(range(shapes[0]), range(shapes[1])):
        img[i * img_rows:i * img_rows + img_rows,
            j * img_cols:j * img_cols + img_cols] = slices[i * shapes[1] + j]
    return img


def natural_sort(l):
    # refer to  https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def array_to_img(img_array, denoise):
    """
    args:
        img_array : 3-d ndarray in [channels, img_rows, img_cols]
        denoise : bool
    return:
        img
    """
    img_array = img_array.transpose((1, 2, 0))
    img = (img_array * 255).astype("uint8")
    if denoise:
        img[img < 128] = 0
        img[img >= 128] = 255
    return img


def tensor_to_img(img_tensor, denoise):
    """
    args:
        img_tensor : 3-d ndarray in [channels, img_rows, img_cols]
        denoise : bool
    return:
        img
    """
    img_array = img_tensor.numpy()
    return array_to_img(img_array, denoise)


def pair_to_rgb(gen_img, tar_img, background='black', use_dilation=False, disk_value=2):
    """
    args:
        gen_img: (ndarray) in [img_rows, img_cols], dytpe=unit8
        tar_img: (ndarray) in [img_rows, img_cols], dytpe=unit8
        background: (str) ['black', 'white']
    return:
        rgb_img: red -> false positive;
                 green -> true positive;
                 blue -> false positive;
    """
    # enhance outline border
    if use_dilation:
        gen_img = dilation(gen_img, disk(disk_value))
        tar_img = dilation(tar_img, disk(disk_value))

    if background == "black":
        # saving rgb results
        rgb_img = np.zeros((gen_img.shape[0], gen_img.shape[1], 3), np.uint8)
        # assign false negative as red channel
        rgb_img[:, :, 0][np.logical_and(gen_img == 255, tar_img == 0)] = 255
        # assign true positive as green channel
        rgb_img[:, :, 1][np.logical_and(gen_img == 255, tar_img == 255)] = 255
        # assign false positive as blue channel
        rgb_img[:, :, 2][np.logical_and(gen_img == 0, tar_img == 255)] = 255
    else:
        # saving rgb results
        rgb_img = np.ones(
            (gen_img.shape[0], gen_img.shape[1], 3), np.uint8) * 255
        # assign false negative as red channel
        rgb_img[:, :, 1][np.logical_and(gen_img == 255, tar_img == 0)] = 0
        rgb_img[:, :, 2][np.logical_and(gen_img == 255, tar_img == 0)] = 0
        # assign true positive as green channel
        rgb_img[:, :, 0][np.logical_and(gen_img == 255, tar_img == 255)] = 0
        rgb_img[:, :, 2][np.logical_and(gen_img == 255, tar_img == 255)] = 0
        # assign false positive as blue channel
        rgb_img[:, :, 0][np.logical_and(gen_img == 0, tar_img == 255)] = 0
        rgb_img[:, :, 1][np.logical_and(gen_img == 0, tar_img == 255)] = 0
    return rgb_img


def shift_edge(img, dtype="uint8"):
    """
    args:
        img : 2-d or 3-d ndarray in [img_rows, img_cols, *channels]
        dtype : unit8 or float32
    return:
        edge: outline of image
    """
    if dtype == "uint8":
        img = (img / 255).astype("float32")
    img_shift_l, img_shift_d = np.copy(img), np.copy(img)
    img_shift_l[:, 1:] = img[:, :-1]
    img_shift_d[1:, :] = img[:-1, :]
    edge_l = img - img_shift_l
    edge_l[edge_l != 0] = 1
    edge_d = img - img_shift_d
    edge_d[edge_d != 0] = 1
    edge = edge_l + edge_d
    edge[edge != 0] = 1
    if dtype == "uint8":
        edge = (edge * 255).astype("uint8")
    return edge


def canny_edge(img, sigma=1):
    """
    args:
        img : 2-d ndarray in [img_rows, img_cols], dtype as unit8
    return:
        edge: outline of image
    """
    edge_bool = feature.canny(img, sigma)
    edge_img = np.zeros((edge_bool.shape), np.uint8)
    edge_img[edge_bool] = 255
    return edge_img


def add_barrier(img, spaces=[2, 5]):
    """
    args:
        img: (ndarray) in [img_rows, img_cols, channels], dtype as unit8
        spaces: (int) pixels of spaces
    return:
        img: (ndarray) processed img
    """
    img = add_color_bar(img, spaces[0], 'black')
    img = add_color_bar(img, spaces[1], 'white')
    return img


def add_color_bar(img, space=5, color='black'):
    """
    args:
        img: (ndarray) in [img_rows, img_cols, channels], dtype as unit8
        space: (int) pixels of space
        color: (str) background color
    return:
        tmp_img: (ndarray) processed img
    """
    if len(img.shape) == 3:
        # adding white space of multi-channel image
        img_rows, img_cols, channels = img.shape
        if color == "white":
            tmp_img = np.ones((img_rows + 2 * space,
                               img_cols + 2 * space,
                               channels), np.uint8) * 255
        elif color == "black":
            tmp_img = np.zeros((img_rows + 2 * space,
                                img_cols + 2 * space,
                                channels), np.uint8)
        else:
            warnings.warn(
                'Required color:{} is not support yet.'.format(color))
            tmp_img = np.ones((img_rows + 2 * space,
                               img_cols + 2 * space,
                               channels), np.uint8) * 128
    else:
        # adding white space of multi-channel image
        img_rows, img_cols = img.shape
        if color == "white":
            tmp_img = np.ones((img_rows + 2 * space,
                               img_cols + 2 * space), np.uint8) * 255
        elif color == "black":
            tmp_img = np.zeros((img_rows + 2 * space,
                                img_cols + 2 * space), np.uint8)
        else:
            warnings.warn(
                'Required color:{} is not support yet.'.format(color))
            tmp_img = np.ones((img_rows + 2 * space,
                               img_cols + 2 * space), np.uint8) * 128
    tmp_img[space: space + img_rows,
            space: space + img_cols] = img
    return tmp_img


def patch_to_img(patches, disp_rows, disp_cols, direction):
    """
      args:
        patches: img patches
        disp_rows: patch number in rows
        disp_cols: patch number in cols
        direction: horizontal or vertical
      return:
        img
    """
    img_rows, img_cols = patches[0].shape[:2]
    if len(patches[0].shape) == 3:
        img = np.zeros((img_rows * disp_rows,
                        img_cols * disp_cols,
                        3), "uint8")
    else:
        img = np.zeros((img_rows * disp_rows,
                        img_cols * disp_cols), "uint8")

    if direction == "horizontal":
        # align models by cols
        for i, j in itertools.product(range(disp_rows), range(disp_cols)):
            img[i * img_rows:(i + 1) * img_rows,
                j * img_cols:(j + 1) * img_cols] = patches[i * disp_cols + j]
    else:
        # align models by rows
        for i, j in itertools.product(range(disp_rows), range(disp_cols)):
            img[j * img_rows:(j + 1) * img_rows,
                i * img_cols:(i + 1) * img_cols] = patches[i * disp_cols + j]

    return img


def three_in_line(imgs, labels):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)
    ax1.imshow(imgs[0])
    ax1.axis('off')
    ax1.set_title(labels[0], fontsize=16)

    ax2.imshow(imgs[1], cmap="gray")
    ax2.axis('off')
    ax2.set_title(labels[1], fontsize=16)

    ax3.imshow(imgs[2], cmap="gray")
    ax3.axis('off')
    ax3.set_title(labels[2], fontsize=16)

    plt.show()


def two_in_line(imgs, labels):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7, 3),
                                   sharex=True, sharey=True)
    ax1.imshow(imgs[0])
    ax1.axis('off')
    ax1.set_title(labels[0], fontsize=16)

    ax2.imshow(imgs[1])
    ax2.axis('off')
    ax2.set_title(labels[1], fontsize=16)

    plt.show()


if __name__ == "__main__":
    # ====================== parameter initialization ======================= #

    # Land-cover,R-value,G-value,B-value
    vaihingen_refs = np.array([
        ["Impervious_surfaces", 255, 255, 255],
        ["Building", 0, 0, 255],
        ["Low vegetation", 0, 255, 255],
        ["Tree", 0, 255, 0],
        ["Car", 255, 255, 0],
        ["Background", 255, 0, 0],
    ])
    # Sample image from Vaihingen-slc Dataset
    sample = 'img_0.png'
    land_img = imread(os.path.join(
        "../dataset/Vaihingen-slc/land", sample))
    segmap_img = imread(os.path.join(
        "../dataset/Vaihingen-slc/segmap", sample))

    two_in_line([land_img, segmap_img],
                ['Land', 'Segmap'])

    label = img_to_label(segmap_img, vaihingen_refs)
    img = label_to_img(label, vaihingen_refs)

    two_in_line([segmap_img, img],
                ['Original', 'Double Convert'])
