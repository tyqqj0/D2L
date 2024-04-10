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
    def __init__(self, r=0.01):
        super(CNLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.num_features = num_features
        self.device = device
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.num_features).to(self.device))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, cn):
        # 计算交叉熵损失
        loss = self.criterion(x, y)
        # 计算中心损失
        center_loss = torch.mean(torch.sum((x - self.centers[y]) ** 2, dim=1))
        # 计算中心更新
        delta_centers = x - self.centers[y]
        for i in range(self.num_classes):
            self.centers[i] += self.alpha * torch.mean(delta_centers[y == i], dim=0)
        # 计算类间距离损失
        inter_loss = 0
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                inter_loss += torch.sum((self.centers[i] - self.centers[j]) ** 2)
        inter_loss /= self.num_classes * (self.num_classes - 1) / 2
        # 计算类内距离损失
        intra_loss = 0
        for i in range(self.num_classes):
            intra_loss += torch.mean(torch.sum((x[y == i] - self.centers[i]) ** 2, dim=1))
        intra_loss /= self.num_classes
        # 计算总损失
        loss += self.beta * center_loss + self.gamma * inter_loss + self.gamma * intra_loss
        return loss



