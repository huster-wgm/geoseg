#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import torch
from torch import nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.criterion = nn.L1Loss(size_average=True)

    def __repr__(self):
        return "L1"

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCELoss(size_average=True)

    def __repr__(self):
        return "BCE"

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss
    
class MCELoss(nn.Module):
    def __init__(self):
        super(MCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(size_average=True)

    def __repr__(self):
        return "MCE"

    def forward(self, output, target):
        target = torch.argmax(target, dim=1).long()
        loss = self.criterion(output, target)
        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
    
    def __repr__(self):
        return "MSE"

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss

class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)

    def __repr__(self):
        return "PSNR"

    def forward(self, output, target):
        mse = self.criterion(output, target)
        loss = 10 * torch.log10(1.0 / mse)
        return loss

    
class AlignLoss(object):
    """
        Object for feature alignments methods
    """

    def __init__(self):
        super(AlignLoss, self).__init__()
        self.criterMSE = nn.MSELoss(size_average=True)
        self.criterBCE = nn.BCELoss(size_average=True)

    def __repr__(self):
        return 'AlignLoss'

    def ALMSE(self, feats):
        zeros = torch.zeros(feats[0].shape)
        if torch.cuda.is_available():
            zeros = zeros.cuda()
        feat_max = None
        feat_min = None
        for feat in feats:
            if feat_max is None:
                feat_max = feat
                feat_min = feat
            else:
                feat_max = torch.max(feat, feat_max)
                feat_min = torch.min(feat, feat_min)
        return self.criterMSE(torch.abs(feat_max - feat_min), zeros)

    def ALBCE(self, feats):
        zeros = torch.zeros(feats[0].shape)
        if torch.cuda.is_available():
            zeros = zeros.cuda()
        feat_max = None
        feat_min = None
        for feat in feats:
            if feat_max is None:
                feat_max = feat
                feat_min = feat
            else:
                feat_max = torch.max(feat, feat_max)
                feat_min = torch.min(feat, feat_min)
        return self.criterBCE(torch.abs(feat_max - feat_min), zeros)
