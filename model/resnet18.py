# -*- CODING: UTF-8 -*-
# @time 2024/1/25 12:01
# @Author tyqqj
# @File resnet18.py
# @
# @Aim
import torch
from torch import nn as nn
from torchvision import models
from torch import flatten  # 对卷积是否正确存疑 TODO

__ALL__ = ['ResNet18FeatureExtractor']

from torchvision.models import ResNet18_Weights


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, in_channels=1):
        super(ResNet18FeatureExtractor, self).__init__()
        # 使用ResNet18模型，设置输入通道数为1，输出类别数为10
        if pretrained:
            # 使用默认的预训练权重
            weights = ResNet18_Weights.DEFAULT
        else:
            # 如果不使用预训练模型，则权重为None
            weights = None

            # 加载ResNet18模型
        self.resnet18 = models.resnet18(weights=weights, num_classes=num_classes)

        # 如果输入通道数不是3，需要修改模型的第一层
        if in_channels != 3:
            # 获取原始第一层的权重
            original_first_layer = self.resnet18.conv1
            # 创建一个新的第一层，输入通道数为in_channels
            new_first_layer = nn.Conv2d(in_channels, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size,
                                        stride=original_first_layer.stride,
                                        padding=original_first_layer.padding,
                                        bias=original_first_layer.bias)

            # 如果是使用预训练权重的话，这里可以复制权重，取平均等方式修改权重
            # 但是这里只是简单地用随机权重初始化
            self.resnet18.conv1 = new_first_layer

        # 如果你需要使用预训练权重，你可能需要手动复制权重从3通道到1通道
        # 这里是一个简单的平均权重的方法，如果你有预训练的权重，那么取消下面两行的注释
        if pretrained:
            conv1_weight = self.resnet18.conv1.weight.data.mean(dim=1, keepdim=True)
            self.resnet18.conv1.weight.data = conv1_weight

    def forward(self, x):
        # 获取softmax之前的特征
        # Use ResNet18 up to the second-to-last layer to get the features
        features = {}
        x = self.resnet18.conv1(x)
        features['conv1'] = flatten(x, 1)  # (16, 64, 14, 14) (batch_size, C, H, W)->(batch_size, C*H*W)
        x = self.resnet18.bn1(x)
        features['bn1'] = flatten(x, 1)
        x = self.resnet18.relu(x)
        features['relu1'] = flatten(x, 1)
        x = self.resnet18.maxpool(x)
        features['maxpool'] = flatten(x, 1)
        x = self.resnet18.layer1(x)
        features['layer1'] = flatten(x, 1)
        x = self.resnet18.layer2(x)
        features['layer2'] = flatten(x, 1)
        x = self.resnet18.layer3(x)
        features['layer3'] = flatten(x, 1)
        x = self.resnet18.layer4(x)
        features['layer4'] = flatten(x, 1)
        x = self.resnet18.avgpool(x)
        # features['avgpool'] = flatten(x, 1)
        x = torch.flatten(x, 1)  # Flatten the features
        # features['flatten'] = flatten(x, 1)
        # Pass the features through the new fully connected layer
        output = self.resnet18.fc(x)  # Make sure new_fc is defined in __init__ and has the correct in_features
        return output, features

    def get_layer_conv_num(self):
        pass
