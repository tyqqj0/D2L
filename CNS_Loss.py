# -*- CODING: UTF-8 -*-
# @time 2024/4/10 11:15
# @Author tyqqj
# @File CNS_Loss.py
# @
# @Aim 

import torch
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

        # self._y_stars = None
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

        # similarity = torch.bmm(cn, hi.t())  # similarity: (k, num, n)

        # if hi.dim() == 2:
        #     hi = hi.unsqueeze(0)  # 在第0维添加一个维度，使其成为三维张量

        # print(cn.shape, hi.t().shape)

        # 对hi进行标准化
        # hi_mean = hi.mean(dim=1, keepdim=True)  # 计算每个特征的均值
        # hi_std = hi.std(dim=1, keepdim=True)  # 计算每个特征的标准差
        # print(hi_mean.shape)
        # hi = (hi - hi_mean) / (hi_std + 1e-8)  # 标准化

        # 对hi进行归一化
        hi_norm = torch.norm(hi, dim=1, keepdim=True)  # 计算每个向量的模长
        hi = hi / (hi_norm + 1e-8)  # 归一化

        # similarity = torch.bmm(cn, hi.transpose(1, 2))  # similarity: (k, num, n)

        similarity = torch.einsum('klm,mn->kln', cn, hi.t())  # similarity: (k, num, n)
        # print("simi:", similarity.shape)
        # e_similarity = torch.exp(similarity)
        # y_stars = e_similarity / torch.sum(e_similarity, dim=2, keepdim=True)  # y_stars: (k, num, n)
        # print(similarity)
        # 对每个聚类(k)找到最相似的特征向量的相似度,得到(k, n)的张量
        y_star, _ = torch.max(similarity, dim=1)  # y_star: (k, n)
        # y* = cn(n, k max(num))
        # 将y_star的维度调整为(n, k)
        y_star = y_star.t()  # y_star: (n, k)
        # return y_star

        # 对y_star进行标准化
        # y_star_mean = y_star.mean(dim=0, keepdim=True)  # 计算每个聚类的均值
        # y_star_std = y_star.std(dim=0, keepdim=True)  # 计算每个聚类的标准差
        # y_star = (y_star - y_star_mean) / (y_star_std + 1e-8)  # 标准化

        return abs(y_star)

    @property
    def _alpha(self):
        # print('aplha')
        # alpha:(1 - 1/(max(y_star*(1-y)) - r + 1))
        # 计算alpha
        y_star = self._y_stars
        # print(y_star.min().item(), y_star.max().item(), y_star.mean().item())
        # print('here')
        # 计算y_star*(1-y)
        y = self.y
        if len(y.shape) == 1:
            # 将y one-hot编码
            y = torch.eye(self.classs).to(y.device)[y]
            # print('here')
        # print(y.shape, y_star.shape)
        y_star_1_y = y_star * (1 - y)
        # print(y_star_1_y.shape)
        # 计算max(y_star*(1-y))
        max_y_star_1_y, _ = torch.max(y_star_1_y, dim=1)
        # print(max_y_star_1_y.shape)
        # print(max_y_star_1_y.min().item(), max_y_star_1_y.max().item(), max_y_star_1_y.mean().item())
        # 计算alpha
        m = torch.max(max_y_star_1_y - self.r, torch.zeros_like(max_y_star_1_y))
        # print(m.min().item(), m.max().item(), m.mean().item())
        alpha = 2 * (1 - (1 / (m + 1))) # (n, 1), 0<=alpha<=1防止有负数出现
        # 整成(n)
        # print(alpha.shape)
        if len(alpha.shape) == 2:
            alpha = alpha.squeeze()#
        alpha = torch.min(alpha, torch.ones_like(alpha))
        return alpha

    def forward(self, x, y, hi):
        # loss=(1-alpha)*cross_entropy + alpha*cn_loss
        self.hi = hi
        self.y = y
        self.x = x
        y_star = self._y_stars
        # print(y_star, y_star.min().item(), y_star.max().item(), y_star.mean().item())
        # print(y_star.min().item(), y_star.max().item(), y_star.mean().item())
        # alpha = self._alpha
        # 计算交叉熵损失
        cross_entropy_loss = self.loss(x, y)
        # print(cross_entropy_loss)

        # print(y.shape, x.shape, y_star.shape)

        # 计算cn损失
        cn_loss = self.cn_loss(x, y_star)
        # print(cn_loss)

        # 计算alpha
        alpha = self._alpha
        print(alpha.min().item(), alpha.max().item(), alpha.mean().item())

        # 计算总损失
        loss = (1 - alpha) * cross_entropy_loss + alpha * cn_loss
        # print(loss.shape)

        return loss.mean()
