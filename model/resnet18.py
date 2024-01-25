# -*- CODING: UTF-8 -*-
# @time 2024/1/25 12:01
# @Author tyqqj
# @File resnet18.py
# @
# @Aim
import torch
from torch import nn as nn
from torchvision import models


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained=False, num_classes=10):
        super(ResNet18FeatureExtractor, self).__init__()
        # 使用ResNet18模型，设置输入通道数为1，输出类别数为10
        self.resnet18 = models.resnet18(pretrained=pretrained, num_classes=num_classes)
        # 修改第一个卷积层为单通道输入
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 如果你需要使用预训练权重，你可能需要手动复制权重从3通道到1通道
        # 这里是一个简单的平均权重的方法，如果你有预训练的权重，那么取消下面两行的注释
        if pretrained:
            conv1_weight = self.resnet18.conv1.weight.data.mean(dim=1, keepdim=True)
            self.resnet18.conv1.weight.data = conv1_weight

    def forward(self, x):
        # 获取softmax之前的特征
        # Use ResNet18 up to the second-to-last layer to get the features
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        features = torch.flatten(x, 1)  # Flatten the features

        # Pass the features through the new fully connected layer
        output = self.resnet18.fc(features)  # Make sure new_fc is defined in __init__ and has the correct in_features
        return output, features
