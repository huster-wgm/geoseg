#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from skimage.io import imread, imsave
from skimage.transform import resize

import vision
import argparse


DIR = os.path.dirname(os.path.abspath(__file__))
Dataset_DIR = os.path.join(DIR, '../dataset/')


class BinaryDataset(data.Dataset):
    """Binary datasets """
    def __init__(
            self, root='NZ32km2', split='all'):
        """
        Args:
            root (str): root dir of dataset
            split (str): part of the dataset ['all', 'train', 'val']
        """
        self.root = root
        self.split = split

        self.srcpath = os.path.join(Dataset_DIR, root, "img", '%s')
        self.tarpath = os.path.join(Dataset_DIR, root, "msk", '%s')
        # get datalist
        self.datalist = pd.read_csv(os.path.join(
            Dataset_DIR, root, '{}.csv'.format(split)))['id'].tolist()

        self.src_ch = 3
        self.tar_ch = 1

    def __len__(self):
        return len(self.datalist)

    def _whitespace(self, img, width=5):
        """
        Args:
            img : ndarray [h,w,c]
        """
        row, col, ch = img.shape
        tmp = np.ones((row + 2*width, col + 2*width, ch), "uint8") * 128
        tmp[width:row+width,width:width+col,:] = img
        return tmp

    def _src2img(self, arr):
        """
        Args:
            arr (str): ndarray [h,w,c]
        """
        img = (arr * 255).astype("uint8")
        return self._whitespace(img)

    def _tar2img(self, arr, x8=False):
        """
        Args:
            arr (str): ndarray [h,w,c]
        """
        assert arr.shape[2] == 1, "Output should be single channel."
        arr[arr < 0.5] = 0
        arr[arr >= 0.5] = 1
        arr = (arr * 255).astype('uint8')
        img = np.concatenate([arr, arr, arr], axis=-1)
        if x8:
            row, col = img.shape[:2]
            img = cv2.resize(
                img, (col * 8, row * 8), interpolation = cv2.INTER_NEAREST)
        return self._whitespace(img)


class BinaryIM(BinaryDataset):
    """Binary datasets with Img(I) and Mask(M)"""
    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = imread(src_file)
        tar = imread(tar_file)
        assert len(tar.shape) == 2, "Mask should be 2D."
        # src => uint8 to float tensor
        src = (src / 255).transpose((2, 0, 1))
        src = torch.from_numpy(src).float()
        # tar => uint8 to float tensor
        tar = np.expand_dims(tar, axis=-1).transpose((2, 0, 1))
        tar = torch.from_numpy(tar).float()
        sample = {"src":src,
                  "tar":tar,}
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        # convert array to RGB img
        src_img = self._src2img(src)
        tar_img = self._tar2img(tar)

        vis_img = self._whitespace(np.concatenate([src_img, tar_img], axis=1))
        save_dir = os.path.join(DIR, "../example", self.root)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}-IM.png".format(save_dir, self.split, idx), vis_img)

        
class BinaryIMS(BinaryDataset):
    """Binary datasets with Img(I), Mask(M) and 1/8 Sub-Mask(S)
       required data format for MC-FCN \
       "Automatic Building Segmentation of Aerial Imagery Using Multi-Constraint Fully Convolutional Networks. \
       Remote Sens. 2018"
    """
    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = imread(src_file)
        tar = imread(tar_file)
        assert len(tar.shape) == 2, "Mask should be 2D."
        img_rows, img_cols = tar.shape[:2]
        tar_sub = cv2.resize(
            tar, (img_cols // 8, img_rows // 8), interpolation = cv2.INTER_NEAREST)
        # src => uint8 to float tensor
        src = (src / 255).transpose((2, 0, 1))
        src = torch.from_numpy(src).float()
        # tar => uint8 to float tensor
        tar = np.expand_dims(tar, axis=-1).transpose((2, 0, 1))
        tar = torch.from_numpy(tar).float()
        # tar_sub => float to float tensor
        tar_sub = np.expand_dims(tar_sub, axis=-1).transpose((2, 0, 1))
        tar_sub = torch.from_numpy(tar_sub).float()
        sample = {"src":src,
                  "tar":tar,
                  "tar_sub":tar_sub,
                 }
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        tar_sub = sample['tar_sub'].numpy().transpose((1, 2, 0))
        # convert array to RGB img
        src_img = self._src2img(src)
        tar_img = self._tar2img(tar)
        tar_sub_img = self._tar2img(tar_sub, True)
        vis_img = self._whitespace(np.concatenate([src_img, tar_img, tar_sub_img], axis=1))
        save_dir = os.path.join(DIR, "../example", self.root)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}-IMS.png".format(save_dir, self.split, idx), vis_img)

        
class BinaryIME(BinaryDataset):
    """Binary datasets with Img(I), Mask(M) and Edge(E)
       required data format for BR-Net \
       "A Boundary Regulated Network for Accurate Roof Segmentation and Outline Extraction \
       Remote Sens. 2018."
    """
    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = imread(src_file)
        tar = imread(tar_file)
        assert len(tar.shape) == 2, "Mask should be 2D."
        tar_sub = vision.shift_edge(tar, self.tar_ch)
        # src => uint8 to float tensor
        src = (src / 255).transpose((2, 0, 1))
        src = torch.from_numpy(src).float()
        # tar => uint8 to float tensor
        tar = np.expand_dims(tar, axis=-1).transpose((2, 0, 1))
        tar = torch.from_numpy(tar).float()
        # tar_sub => float to float tensor
        tar_sub = tar_sub.transpose((2, 0, 1))
        tar_sub = torch.from_numpy(tar_sub).float()
        sample = {"src":src,
                  "tar":tar,
                  "tar_sub":tar_sub,
                 }
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        tar_sub = sample['tar_sub'].numpy().transpose((1, 2, 0))
        # convert array to RGB img
        src_img = self._src2img(src)
        tar_img = self._tar2img(tar)
        tar_sub_img = self._tar2img(tar_sub)
        vis_img = self._whitespace(np.concatenate([src_img, tar_img, tar_sub_img], axis=1))
        save_dir = os.path.join(DIR, "../example", self.root)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}-IME.png".format(save_dir, self.split, idx), vis_img)


class MultiDataset(data.Dataset):
    """Multi-label datasets """
    def __init__(
            self, root='Vaihingen', split='all'):
        """
        Args:
            root (str): root dir of dataset
            split (str): part of the dataset ['all', 'train', 'val']
        """
        self.root = root
        self.split = split

        self.srcpath = os.path.join(Dataset_DIR, root, "img", '%s')
        self.tarpath = os.path.join(Dataset_DIR, root, "msk", '%s')
        # get datalist
        self.datalist = pd.read_csv(os.path.join(
            Dataset_DIR, root, '{}.csv'.format(split)))['id'].tolist()
        self.ref = pd.read_csv(os.path.join(
            Dataset_DIR, root, 'ref.csv'))
        self.cmap = (self.ref.iloc[:,1:]).values
        self.src_ch = 4 if 'RGBIR' in root else 3
        self.tar_ch = self.ref.shape[0]

    def __len__(self):
        return len(self.datalist)

    def _whitespace(self, img, width=5):
        """
        Args:
            img : ndarray [h,w,c]
        """
        row, col, ch = img.shape
        tmp = np.ones((row + 2*width, col + 2*width, ch), "uint8") * 128
        tmp[width:row+width,width:width+col,:] = img
        return tmp

    def _src2img(self, arr):
        """
        Args:
            arr (str): ndarray [h,w,c]
        """
        img = (arr * 255).astype("uint8")
        return self._whitespace(img)

    def _tar2img(self, arr, x8=False):
        """
        Args:
            arr (str): ndarray [h,w,c]
        """
        assert arr.shape[2] > 1, "Output should be multi channel."
        img = vision.label_to_img(arr, self.cmap)
        if x8:
            row, col = img.shape[:2]
            img = cv2.resize(
                img, (col * 8, row * 8), interpolation = cv2.INTER_NEAREST)
        return self._whitespace(img)


class MultiIM(MultiDataset):
    """Multi-label datasets with Img(I) and Mask(M)"""
    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = imread(src_file)
        tar = imread(tar_file)
        assert len(tar.shape) == 2, "Mask should be 2D."
        # src => uint8 to float tensor
        src = (src / 255).transpose((2, 0, 1))
        src = torch.from_numpy(src).float()
        # tar => uint8 to float tensor
        tar = vision.cls_to_label(tar, self.tar_ch).transpose((2, 0, 1))
        tar = torch.from_numpy(tar).float()
        sample = {"src":src,
                  "tar":tar,}
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        # convert array to RGB img
        src_img = self._src2img(src)
        tar_img = self._tar2img(tar)

        vis_img = self._whitespace(np.concatenate([src_img, tar_img], axis=1))
        save_dir = os.path.join(DIR, "../example", self.root)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}-IM.png".format(save_dir, self.split, idx), vis_img)

        
class MultiIMS(MultiDataset):
    """Multi-label datasets with Img(I), Mask(M) and 1/8 Sub-Mask(S)
       required data format for MC-FCN \
       "Automatic Building Segmentation of Aerial Imagery Using Multi-Constraint Fully Convolutional Networks. \
       Remote Sens. 2018"
    """
    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = imread(src_file)
        tar = imread(tar_file)
        assert len(tar.shape) == 2, "Mask should be 2D."
        img_rows, img_cols = tar.shape[:2]
        tar_sub = cv2.resize(
            tar, (img_cols // 8, img_rows // 8), interpolation = cv2.INTER_NEAREST)
        # src => uint8 to float tensor
        src = (src / 255).transpose((2, 0, 1))
        src = torch.from_numpy(src).float()
        # tar => uint8 to float tensor
        tar = vision.cls_to_label(tar, self.tar_ch).transpose((2, 0, 1))
        tar = torch.from_numpy(tar).float()
        # tar_sub => float to float tensor
        tar_sub = vision.cls_to_label(tar_sub, self.tar_ch).transpose((2, 0, 1))
        tar_sub = torch.from_numpy(tar_sub).float()
        sample = {"src":src,
                  "tar":tar,
                  "tar_sub":tar_sub,
                 }
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        tar_sub = sample['tar_sub'].numpy().transpose((1, 2, 0))
        # convert array to RGB img
        src_img = self._src2img(src)
        tar_img = self._tar2img(tar)
        tar_sub_img = self._tar2img(tar_sub, True)
        vis_img = self._whitespace(np.concatenate([src_img, tar_img, tar_sub_img], axis=1))
        save_dir = os.path.join(DIR, "../example", self.root)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}-IMS.png".format(save_dir, self.split, idx), vis_img)

        
class MultiIME(MultiDataset):
    """Multi-label datasets with Img(I), Mask(M) and Edge(E)
       required data format for BR-Net \
       "A Boundary Regulated Network for Accurate Roof Segmentation and Outline Extraction \
       Remote Sens. 2018."
    """
    def __getitem__(self, idx):
        src_file = self.srcpath % self.datalist[idx]
        tar_file = self.tarpath % self.datalist[idx]

        src = imread(src_file)
        tar = imread(tar_file)
        assert len(tar.shape) == 2, "Mask should be 2D."
        tar_sub = vision.shift_edge(tar, self.tar_ch)
        # src => uint8 to float tensor
        src = (src / 255).transpose((2, 0, 1))
        src = torch.from_numpy(src).float()
        # tar => uint8 to float tensor
        tar = vision.cls_to_label(tar, self.tar_ch).transpose((2, 0, 1))
        tar = torch.from_numpy(tar).float()
        # tar_sub => float to float tensor
        tar_sub = tar_sub.transpose((2, 0, 1))
        tar_sub = torch.from_numpy(tar_sub).float()
        sample = {"src":src,
                  "tar":tar,
                  "tar_sub":tar_sub,
                 }
        return sample

    def show(self, idx):
        sample = self.__getitem__(idx)
        # tensor to array
        src = sample['src'].numpy().transpose((1, 2, 0))
        tar = sample['tar'].numpy().transpose((1, 2, 0))
        tar_sub = sample['tar_sub'].numpy().transpose((1, 2, 0))
        # convert array to RGB img
        src_img = self._src2img(src)
        tar_img = self._tar2img(tar)
        tar_sub_img = self._tar2img(tar_sub)
        vis_img = self._whitespace(np.concatenate([src_img, tar_img, tar_sub_img], axis=1))
        save_dir = os.path.join(DIR, "../example", self.root)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave("{}/{}-{}-IME.png".format(save_dir, self.split, idx), vis_img)

        
def load_dataset(root, mode):
    """
    Args:
        root (str): root of dataset
        mode (str): ['IM', 'IMS', 'IME'] of dataset
    """
    
    if root == "NZ32km2":
        version = "Binary"
    else:
        version = "Multi"
    version += mode
    # setup dataset
    trainset = eval(version)(root=root, split="train")
    valset = eval(version)(root=root, split="val")
    return trainset, valset


if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-idx', type=int, default=0,
                        help='index of sample image')
    args = parser.parse_args()
    idx = args.idx
    for root in ['NZ32km2', 'Vaihingen', 'PotsdamRGB', 'PotsdamRGB', 'PotsdamIRRG']:
        for mode in ["IM", "IMS", "IME"]:
            print("Load {}/{}.".format(root, mode))
            trainset, valset = load_dataset(root, mode)
            
            # print("Load train set = {} examples, val set = {} examples".format(
            #     len(trainset), len(valset)))
            sample = trainset[idx]
            trainset.show(idx)
            sample = valset[idx]
            valset.show(idx)
            print("\tsrc:", sample["src"].shape,
                  "tar:", sample["tar"].shape,)

