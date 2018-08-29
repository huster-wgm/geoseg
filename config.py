#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
from utils.datasets import *


class LSdataset(object):
    """
        Dataset setting for NewZealand
    """

    def __init__(self):
        self.train = nzLS("train")
        self.val = nzLS("val")
        self.test = nzLS("test")
        self.in_ch = 3
        self.out_ch = self.train.nb_class


class LSSubdataset(object):
    """
        Dataset setting for NewZealand
    """

    def __init__(self):
        self.train8xsub = nzLS8xsub("train")
        self.train = nzLS("train")
        self.val = nzLS("val")
        self.test = nzLS("test")
        self.in_ch = 3
        self.out_ch = self.train.nb_class


class LSEdataset(object):
    """
        Dataset setting for NewZealand
    """

    def __init__(self):
        self.trainLSE = nzLSE("train")
        self.train = nzLS("train")
        self.val = nzLS("val")
        self.test = nzLS("test")
        self.in_ch = 3
        self.out_ch = self.train.nb_class


if __name__ == "__main__":
    print("Hello world")
