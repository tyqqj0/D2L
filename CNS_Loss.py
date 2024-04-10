# -*- CODING: UTF-8 -*-
# @time 2024/4/10 11:15
# @Author tyqqj
# @File CNS_Loss.py
# @
# @Aim 

import numpy as np
import torch
from torch.utils import data
from torch import nn


# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision


class CNLoss(nn.Module):
    def __init__(self, loss=''):
        super(CNLoss, self).__init__()
        if loss == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        # 交叉熵来计算这个
        return self.loss(x, y)


class CNSLosst(nn.Module):
    def __init__(self, classs=10, r=0.01, loss=''):
        super(CNSLosst, self).__init__()
        self.y = None
        self.r = r
        if loss == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()

        self._y_stars = None
        self.hi = None
        self.cn = None
        self.cn_loss = CNLoss(loss=loss)
        self.classs = classs

    def set_cn(self, cn):
        self.cn = cn

    @property
    def _y_stars(self):
        # hi:(n, C, h, w), cn(k, num, M)
        hi = self.hi
        cn = self.cn
        if hi is None:
            raise ValueError('hi is None')

        if len(hi.shape) == 4:
            hi = hi.view(hi.shape[0], -1)
            # (n, M)
        if cn is None:
            # (k, 1, M*0)
            # cn = torch.zeros(self.classs, 1, hi.shape[1]).to(hi.device)
            # 直接返回全零即可(n, k*0)
            return torch.zeros(hi.shape[0], self.classs).to(hi.device)

        # hi*cn -> (n, k, num)
        # 计算cn和hi的转置的乘积,得到(k, num, n)的张量
        similarity = torch.matmul(cn, hi.t())  # similarity: (k, num, n)
        # 对每个聚类(k)找到最相似的特征向量的相似度,得到(k, n)的张量
        y_star, _ = torch.max(similarity, dim=1)  # y_star: (k, n)
        # y* = cn(n, k max(num))
        # 将y_star的维度调整为(n, k)
        y_star = y_star.t()  # y_star: (n, k)
        # return y_star

        return y_star

    @property
    def _alpha(self):
        # alpha:(1 - 1/(max(y_star*(1-y)) - r + 1))
        # 计算alpha
        y_star = self._y_stars
        # 计算y_star*(1-y)
        y = self.y
        y_star_1_y = y_star * (1 - y)
        # 计算max(y_star*(1-y))
        max_y_star_1_y, _ = torch.max(y_star_1_y, dim=1)
        # 计算alpha
        alpha = 1 - 1 / (max(max_y_star_1_y - self.r, 0) + 1)  # (n, 1), 0<=alpha<=1防止有负数出现
        # 整成(n)
        if len(alpha.shape) == 2:
            alpha = alpha.squeeze()
        return alpha

    def forward(self, x, hi, y):
        # loss=(1-alpha)*cross_entropy + alpha*cn_loss
        self.hi = hi
        self.y = y
        # 计算交叉熵损失
        cross_entropy_loss = self.loss(x, y)

        # 计算cn损失
        cn_loss = self.cn_loss(x, self._y_stars)

        # 计算alpha
        alpha = self._alpha

        # 计算总损失
        loss = (1 - alpha) * cross_entropy_loss + alpha * cn_loss

        return loss
