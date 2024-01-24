# -*- CODING: UTF-8 -*-
# @time 2024/1/24 19:39
# @Author tyqqj
# @File main.py
# @
# @Aim

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms
from torchvision import models
import numpy as np

import mlflow
from LID import mle_batch_np


class NoisyMNIST(Dataset):
    def __init__(self, mnist_dataset, noise_ratio=0.0, noise_type='sym'):
        self.mnist_dataset = mnist_dataset
        self.noise_ratio = noise_ratio
        self.noise_type = noise_type
        self.apply_noise()

    def apply_noise(self):
        if self.noise_ratio > 0:
            n_samples = len(self.mnist_dataset)
            n_noisy = int(self.noise_ratio * n_samples)
            indices = np.random.choice(n_samples, n_noisy, replace=False)
            for idx in indices:
                if self.noise_type == 'sym':
                    # 对称噪声：随机选择一个不同的标签
                    self.mnist_dataset.targets[idx] = torch.randint(0, 10, (1,)).item()
                # 这里可以添加其他类型的噪声处理
                # ...

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        return self.mnist_dataset[idx]


def load_data(path='D:/gkw/data/classification', dataset_name='MNIST', batch_size=128, noise_ratio=0.0,
              noise_type='sym'):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 原始MNIST数据集
        mnist_train = datasets.MNIST(root=path, train=True, transform=transform, download=True)
        mnist_test = datasets.MNIST(root=path, train=False, transform=transform)

        # 应用噪声
        train_dataset = NoisyMNIST(mnist_train, noise_ratio=noise_ratio, noise_type=noise_type)
        test_dataset = NoisyMNIST(mnist_test)  # 测试集通常不添加噪声

        # 数据加载器
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained=False, num_classes=10):
        super(ResNet18FeatureExtractor, self).__init__()
        # 使用预训练的ResNet18模型，但不包括最后的全连接层
        self.resnet18 = models.resnet18(pretrained=pretrained)
        self.resnet18.fc = nn.Identity()  # 将最后的全连接层替换为一个恒等映射
        self.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        # 获取softmax之前的特征
        features = self.resnet18(x)
        # 通过新的全连接层得到最终的输出
        output = self.fc(features)
        return output, features


def text_in_box(text, length=65, center=True):
    # Split the text into lines that are at most `length` characters long
    lines = [text[i:i + length] for i in range(0, len(text), length)]

    # Create the box border, with a width of `length` characters
    up_border = '┏' + '━' * (length + 2) + '┓'
    down_border = '┗' + '━' * (length + 2) + '┛'
    # Create the box contents
    contents = '\n'.join(['┃ ' + (line.center(length) if center else line.ljust(length)) + ' ┃' for line in lines])

    # Combine the border and contents to create the final box
    box = '\n'.join([up_border, contents, down_border])

    return box


def get_lids_random_batch(model, dataloader, k=20):
    lids = []

    for X_batch in dataloader:
        if model is None:
            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch = mle_batch_np(X_batch, X_batch, k=k)
        else:
            # Get deep representations
            X_act = model(X_batch)  # Assuming the model is a PyTorch model
            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch = mle_batch_np(X_act, X_act, k=k)

        lids.extend(lid_batch)

    lids = np.asarray(lids, dtype=np.float32)
    return np.mean(lids), np.std(lids)


def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    logits_list = []

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, logits = model(inputs)
        logits_list.append(logits)
        loss = criterion(outputs, targets)
        print("loss:", loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    train_loss = running_loss / len(data_loader)
    train_accuracy = 100. * correct / total
    lid_mean, lid_std = get_lids_random_batch(model, data_loader)
    return train_loss, train_accuracy, lid_mean, lid_std


def val_epoch(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    logits_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, logits = model(inputs)
            logits_list.append(logits)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    val_loss = running_loss / len(data_loader)
    val_accuracy = 100. * correct / total
    lid_mean, lid_std = get_lids_random_batch(model, data_loader)
    return val_loss, val_accuracy, lid_mean, lid_std


def train(model, train_loader, test_loader, optimizer, criterion, scheduler, device, args):
    for epoch in range(args.epochs):
        print(text_in_box('Epoch: %d/%d' % (epoch + 1, args.epochs)))
        train_loss, train_accuracy, train_lid_mean, train_lid_std = train_epoch(model, train_loader, optimizer,
                                                                                criterion,
                                                                                device)
        val_loss, val_accuracy, val_lid_mean, val_lid_std = val_epoch(model, test_loader, criterion, device)

        scheduler.step()

        # 打印训练信息

        print('train_loss: %.3f, train_accuracy: %.3f, train_lid_mean: %.3f, train_lid_std: %.3f' %
              (train_loss, train_accuracy, train_lid_mean, train_lid_std))
        print('val_loss: %.3f, val_accuracy: %.3f, val_lid_mean: %.3f, val_lid_std: %.3f' %
              (val_loss, val_accuracy, val_lid_mean, val_lid_std))

        # mlflow记录
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
        mlflow.log_metric("train_lid_mean", train_lid_mean, step=epoch)
        mlflow.log_metric("train_lid_std", train_lid_std, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        mlflow.log_metric("val_lid_mean", val_lid_mean, step=epoch)
        mlflow.log_metric("val_lid_std", val_lid_std, step=epoch)

    # MLflow记录参数
    mlflow.log_params({
        'epoch': epoch,
        'lr': scheduler.get_last_lr()[0],
        'momentum': args.momentum,
        'weight_decay': args.weight_decay
    })
    # MLflow记录模型
    if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs:
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')
        mlflow.log_artifact(f'model_epoch_{epoch + 1}.pth')


def main(args):
    # 设置mlflow
    mlflow.set_tracking_uri("http://localhost:5002")
    mlflow.set_experiment("LID")
    # 获取数据集
    train_loader, test_loader = load_data(path='D:/gkw/data/classification', dataset_name=args.dataset,
                                          batch_size=args.batch_size, noise_ratio=args.noise_ratio,
                                          noise_type=args.noise_type)
    print("train_loader:", train_loader)
    print("test_loader:", test_loader)
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # 设置模型
    if args.model == 'resnet18':
        model = ResNet18FeatureExtractor(pretrained=False, num_classes=args.num_classes)
    else:
        raise NotImplementedError('model not implemented!')

    # 设置优化器
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    # 设置学习率调整策略
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    mlflow.start_run(run_name=args.run_name)
    # 训练
    train(model, train_loader, test_loader, optimizer, criterion, scheduler, device, args)
    mlflow.end_run()


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='PyTorch ResNet with LID')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100/SVHN]')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--noise_ratio', default=0.0, type=float, help='corruption ratio, should be less than 1')
    parser.add_argument('--noise_type', default='sym', type=str, help='[sym/asym]')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--k', default=50, type=int, help='number of nearest neighbors')
    parser.add_argument('--model', default='resnet18', type=str, help='model type')
    parser.add_argument('--exp_name', default='train_LID', type=str, help='exp name')
    parser.add_argument('--run_name', default='run_1', type=str, help='run name')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    main(args)
