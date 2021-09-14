# -*- coding: utf-8 -*-

'''
@Time    : 2021/7/22 15:14
@Author  : Qiushi Wang
@FileName: focalloss.py
@Software: PyCharm
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.0, alpha=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, inputs, targets):
        loss = self.criterion(inputs, targets)

        return torch.mul(loss, torch.pow((1 - torch.exp(-1*loss)), self.gamma))
