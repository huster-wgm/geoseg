#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com \
           guozhilingty@gmail.com
  @Copyright: go-hiroaki & Chokurei
  @License: MIT
"""
import torch

esp = 1e-5


def _binarize(y_data, threshold):
    """
    args:
        y_data : [float] 4-d tensor in [batch_size, channels, img_rows, img_cols]
        threshold : [float] [0.0, 1.0]
    return 3-d binarized [int] y_data
    """
    y_data[y_data < threshold] = 0.0
    y_data[y_data >= threshold] = 1.0
    return y_data[:,0,:,:].int()


def _argmax(y_data, dim):
    """
    args:
        y_data : 4-d tensor in [batch_size, channels, img_rows, img_cols]
        dim : int
    return 3-d [int] y_data
    """
    return torch.argmax(y_data, dim).int()


def _get_tp(y_pred, y_true):
    """
    args:
        y_true : [int] 3-d in [batch_size, img_rows, img_cols]
        y_pred : [int] 3-d in [batch_size, img_rows, img_cols]
    return [float] true_positive
    """
    return torch.sum(y_true * y_pred).float()


def _get_fp(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] false_positive
    """
    return torch.sum((1 - y_true) * y_pred).float()


def _get_tn(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] true_negative
    """
    return torch.sum((1 - y_true) * (1 - y_pred)).float()


def _get_fn(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] false_negative
    """
    return torch.sum(y_true * (1 - y_pred)).float()


def confusion_matrix(y_pred, y_true, dim=1, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return confusion matrix
    """
    batch_size, channels, img_rows, img_cols = y_true.shape
    if channels == 1:
        y_pred = _binarize(y_pred, threshold)
        y_true = _binarize(y_true, threshold)
    else:
        y_pred = _argmax(y_pred, dim)
        y_true = _argmax(y_true, dim)
    performs = torch.zeros(channels, 4)
    labels = torch.unique(y_true).flip(0)
    labels = labels[labels <= channels]
    for label in labels:
        y_true_l = torch.zeros(batch_size, img_rows, img_cols)
        y_pred_l = torch.zeros(batch_size, img_rows, img_cols)
        y_true_l[y_true == label] = 1
        y_pred_l[y_pred == label] = 1
        nb_tp = _get_tp(y_pred_l, y_true_l)
        nb_fp = _get_fp(y_pred_l, y_true_l)
        nb_tn = _get_tn(y_pred_l, y_true_l)
        nb_fn = _get_fn(y_pred_l, y_true_l)
        performs[int(label), :] = [nb_tp, nb_fp, nb_tn, nb_fn]
    mperforms = torch.sum(performs, 0) / len(labels)
    return mperforms, performs


def overall_accuracy(y_pred, y_true, dim=1, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return (tp+tn)/total
    """
    batch_size, channels, img_rows, img_cols = y_true.shape
    if channels == 1:
        y_pred = _binarize(y_pred, threshold)
        y_true = _binarize(y_true, threshold)
    else:
        y_pred = _argmax(y_pred, dim)
        y_true = _argmax(y_true, dim)

    nb_tp_tn = torch.sum(y_true == y_pred).float()
    mperforms = nb_tp_tn / (batch_size * img_rows * img_cols)
    performs = None
    return mperforms, performs


def precision(y_pred, y_true, dim=1, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return tp/(tp+fp)
    """
    batch_size, channels, img_rows, img_cols = y_true.shape
    if channels == 1:
        y_pred = _binarize(y_pred, threshold)
        y_true = _binarize(y_true, threshold)
    else:
        y_pred = _argmax(y_pred, dim)
        y_true = _argmax(y_true, dim)
    performs = torch.zeros(channels, 1)
    labels = torch.unique(y_true).flip(0)
    labels = labels[labels <= channels]
    for label in labels:
        y_true_l = torch.zeros(batch_size, img_rows, img_cols)
        y_pred_l = torch.zeros(batch_size, img_rows, img_cols)
        y_true_l[y_true == label] = 1
        y_pred_l[y_pred == label] = 1
        nb_tp = _get_tp(y_pred_l, y_true_l)
        nb_fp = _get_fp(y_pred_l, y_true_l)
        performs[int(label)] = nb_tp / (nb_tp + nb_fp + esp)
    mperforms = torch.sum(performs, 0) / len(labels)
    return mperforms, performs


def recall(y_pred, y_true, dim=1, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return tp/(tp+fn)
    """
    batch_size, channels, img_rows, img_cols = y_true.shape
    if channels == 1:
        y_pred = _binarize(y_pred, threshold)
        y_true = _binarize(y_true, threshold)
    else:
        y_pred = _argmax(y_pred, dim)
        y_true = _argmax(y_true, dim)
    performs = torch.zeros(channels, 1)
    labels = torch.unique(y_true).flip(0)
    labels = labels[labels <= channels]
    for label in labels:
        y_true_l = torch.zeros(batch_size, img_rows, img_cols)
        y_pred_l = torch.zeros(batch_size, img_rows, img_cols)
        y_true_l[y_true == label] = 1
        y_pred_l[y_pred == label] = 1
        nb_tp = _get_tp(y_pred_l, y_true_l)
        nb_fn = _get_fn(y_pred_l, y_true_l)
        performs[int(label)] = nb_tp / (nb_tp + nb_fn + esp)
    mperforms = torch.sum(performs, 0) / len(labels)
    return mperforms, performs


def f1_score(y_pred, y_true, dim=1, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return 2*precision*recall/(precision+recall)
    """
    batch_size, channels, img_rows, img_cols = y_true.shape
    if channels == 1:
        y_pred = _binarize(y_pred, threshold)
        y_true = _binarize(y_true, threshold)
    else:
        y_pred = _argmax(y_pred, dim)
        y_true = _argmax(y_true, dim)
    performs = torch.zeros(channels, 1)
    labels = torch.unique(y_true).flip(0)
    labels = labels[labels <= channels]
    for label in labels:
        y_true_l = torch.zeros(batch_size, img_rows, img_cols)
        y_pred_l = torch.zeros(batch_size, img_rows, img_cols)
        y_true_l[y_true == label] = 1
        y_pred_l[y_pred == label] = 1
        nb_tp = _get_tp(y_pred_l, y_true_l)
        nb_fp = _get_fp(y_pred_l, y_true_l)
        nb_fn = _get_fn(y_pred_l, y_true_l)
        _precision = nb_tp / (nb_tp + nb_fp + esp)
        _recall = nb_tp / (nb_tp + nb_fn + esp)
        performs[int(label)] = 2 * _precision * \
            _recall / (_precision + _recall + esp)
    mperforms = torch.sum(performs, 0) / len(labels)
    return mperforms, performs


def kappa(y_pred, y_true, dim=1, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return (Po-Pe)/(1-Pe)
    """
    batch_size, channels, img_rows, img_cols = y_true.shape
    if channels == 1:
        y_pred = _binarize(y_pred, threshold)
        y_true = _binarize(y_true, threshold)
    else:
        y_pred = _argmax(y_pred, dim)
        y_true = _argmax(y_true, dim)
    performs = torch.zeros(channels, 1)
    labels = torch.unique(y_true).flip(0)
    labels = labels[labels <= channels]
    for label in labels:
        y_true_l = torch.zeros(batch_size, img_rows, img_cols)
        y_pred_l = torch.zeros(batch_size, img_rows, img_cols)
        y_true_l[y_true == label] = 1
        y_pred_l[y_pred == label] = 1
        nb_tp = _get_tp(y_pred_l, y_true_l)
        nb_fp = _get_fp(y_pred_l, y_true_l)
        nb_tn = _get_tn(y_pred_l, y_true_l)
        nb_fn = _get_fn(y_pred_l, y_true_l)
        nb_total = nb_tp + nb_fp + nb_tn + nb_fn
        Po = (nb_tp + nb_tn) / nb_total
        Pe = ((nb_tp + nb_fp) * (nb_tp + nb_fn)
              + (nb_fn + nb_tn) * (nb_fp + nb_tn)) / (nb_total**2)
        performs[int(label)] = (Po - Pe) / (1 - Pe + esp)
    mperforms = torch.sum(performs, 0) / len(labels)
    return mperforms, performs


def jaccard(y_pred, y_true, dim=1, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return intersection / (sum-intersection)
    """
    batch_size, channels, img_rows, img_cols = y_true.shape
    if channels == 1:
        y_pred = _binarize(y_pred, threshold)
        y_true = _binarize(y_true, threshold)
    else:
        y_pred = _argmax(y_pred, dim)
        y_true = _argmax(y_true, dim)
    performs = torch.zeros(channels, 1)
    labels = torch.unique(y_true).flip(0)
    labels = labels[labels <= channels]
    for label in labels:
        y_true_l = torch.zeros(batch_size, img_rows, img_cols)
        y_pred_l = torch.zeros(batch_size, img_rows, img_cols)
        y_true_l[y_true == label] = 1
        y_pred_l[y_pred == label] = 1
        _intersec = torch.sum(y_true_l * y_pred_l).float()
        _sum = torch.sum(y_true_l + y_pred_l).float()
        performs[int(label)] = _intersec / (_sum - _intersec + esp)
    mperforms = torch.sum(performs, 0) / len(labels)
    return mperforms, performs


def create_fake_data(batch_size, channels, img_rows, img_cols):
    """
    args:
        batch_size : int
        channels : int
        img_rows : int
        img_cols : int
    return y_pred_fake, y_true_fake
    """
    pixel = img_cols // channels
    border = 1
    y_true = torch.zeros(batch_size, img_rows, img_cols)
    for i in range(channels):
        y_true[:, :, i*pixel:(i+1)*pixel] = i
    y_pred = y_true.clone()
    y_pred[:, :border, :] = 0
    y_pred[:, img_rows-border:, :] = 0

    y_true_fake = torch.zeros(batch_size, channels, img_rows, img_cols)
    y_pred_fake = torch.zeros(batch_size, channels, img_rows, img_cols)
    for i in range(channels):
        y_true_fake[:,i,:,:][y_true==i] = 1.0
        y_pred_fake[:,i,:,:][y_pred==i] = 1.0
    return y_pred_fake, y_true_fake


if __name__ == "__main__":
    batch_size, channels, img_rows, img_cols = 1, 2, 5, 5
    y_pred, y_true = create_fake_data(batch_size, channels, img_rows, img_cols)
    # print(y_true.shape, torch.argmax(y_true, 1))
    # print(y_pred.shape, torch.argmax(y_pred, 1))

    maccu, accu = overall_accuracy(y_pred, y_true)
    print('mAccu:', maccu, 'Accu', accu)

    mprec, prec = precision(y_pred, y_true)
    print('mPrec:', mprec, 'Prec', prec)

    mreca, reca = recall(y_pred, y_true)
    print('mReca:', mreca, 'Reca', reca)

    mf1sc, f1sc = f1_score(y_pred, y_true)
    print('mF1sc:', mf1sc, 'F1sc', f1sc)

    mkapp, kapp = kappa(y_pred, y_true)
    print('mKapp:', mkapp, 'Kapp', kapp)

    mjacc, jacc = jaccard(y_pred, y_true)
    print('mJacc:', mjacc, 'Jacc', jacc)
