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


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, in_channels=1):
        super(ResNet50FeatureExtractor, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained, num_classes=num_classes)
        if in_channels != 3:
            self.resnet50.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

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
