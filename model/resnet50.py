# -*- CODING: UTF-8 -*-
# @time 2024/3/20 10:16
# @Author tyqqj
# @File resnet50.py
# @
# @Aim

import torch
from torch import nn as nn
from torchvision import models
from torch import flatten  # 对卷积是否正确存疑 TODO

__ALL__ = ['ResNet50FeatureExtractor']

from torchvision.models import ResNet50_Weights


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, in_channels=1):
        super(ResNet50FeatureExtractor, self).__init__()
        if pretrained:
            # 使用默认的预训练权重
            weights = ResNet50_Weights.DEFAULT
        else:
            # 如果不使用预训练模型，则权重为None
            weights = None

            # 加载ResNet18模型
        self.resnet50 = models.resnet50(weights=weights, num_classes=num_classes)

        # 如果输入通道数不是3，需要修改模型的第一层
        if in_channels != 3:
            # 获取原始第一层的权重
            original_first_layer = self.resnet50.conv1
            # 创建一个新的第一层，输入通道数为in_channels
            new_first_layer = nn.Conv2d(in_channels, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size,
                                        stride=original_first_layer.stride,
                                        padding=original_first_layer.padding,
                                        bias=original_first_layer.bias)

            # 如果是使用预训练权重的话，这里可以复制权重，取平均等方式修改权重
            # 但是这里只是简单地用随机权重初始化
            self.resnet50.conv1 = new_first_layer

        if pretrained:
            conv1_weight = self.resnet50.conv1.weight.data.mean(dim=1, keepdim=True)
            self.resnet50.conv1.weight.data = conv1_weight

    def forward(self, x):
        features = {}
        x = self.resnet50.conv1(x)
        features['conv1'] = torch.flatten(x, 1)
        x = self.resnet50.bn1(x)
        features['bn1'] = torch.flatten(x, 1)
        x = self.resnet50.relu(x)
        features['relu1'] = torch.flatten(x, 1)
        x = self.resnet50.maxpool(x)
        features['maxpool'] = torch.flatten(x, 1)
        x = self.resnet50.layer1(x)
        features['layer1'] = torch.flatten(x, 1)
        x = self.resnet50.layer2(x)
        features['layer2'] = torch.flatten(x, 1)
        x = self.resnet50.layer3(x)
        features['layer3'] = torch.flatten(x, 1)
        x = self.resnet50.layer4(x)
        features['layer4'] = torch.flatten(x, 1)
        x = self.resnet50.avgpool(x)
        # features['avgpool'] = torch.flatten(x, 1)
        x = torch.flatten(x, 1)
        # features['flatten'] = torch.flatten(x, 1)
        output = self.resnet50.fc(x)
        return output, features
