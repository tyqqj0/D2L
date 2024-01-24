# -*- CODING: UTF-8 -*-
# @time 2024/1/24 19:39
# @Author tyqqj
# @File main.py
# @
# @Aim

import argparse
import os

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision import models
from tqdm import tqdm

from LID import mle_batch_np


class NoisyMNIST(Dataset):
    def __init__(self, mnist_dataset, noise_ratio=0.0, noise_type='sym'):
        self.mnist_dataset = mnist_dataset
        self.noise_ratio = noise_ratio
        self.noise_type = noise_type

        # 检查是否是Subset，如果是，则直接使用原始数据集的targets属性
        if isinstance(self.mnist_dataset, Subset):
            self.targets = self.mnist_dataset.dataset.targets
        else:
            self.targets = self.mnist_dataset.targets

        self.apply_noise()

    def apply_noise(self):
        if self.noise_ratio > 0:
            n_samples = len(self.mnist_dataset)
            n_noisy = int(self.noise_ratio * n_samples)
            indices = np.random.choice(n_samples, n_noisy, replace=False)
            for idx in indices:
                if self.noise_type == 'sym':
                    # 生成新的随机标签
                    new_label = torch.randint(0, 10, (1,)).item()
                    # 确保新标签与原标签不同
                    while new_label == self.targets[idx]:
                        new_label = torch.randint(0, 10, (1,)).item()
                    # 应用噪声
                    self.targets[idx] = new_label

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        if isinstance(self.mnist_dataset, Subset):
            # 如果是Subset，则从原始数据集中获取数据和标签
            img, _ = self.mnist_dataset.dataset[self.mnist_dataset.indices[idx]]
            target = self.targets[self.mnist_dataset.indices[idx]]
        else:
            # 如果不是Subset，则直接从数据集中获取数据和标签
            img, target = self.mnist_dataset[idx]

        return img, target


def load_data(path='D:/gkw/data/classification', max_data=1024, dataset_name='MNIST', batch_size=128, noise_ratio=0.0,
              noise_type='sym'):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 原始MNIST数据集
        mnist_train = datasets.MNIST(root=path, train=True, transform=transform, download=True)
        mnist_test = datasets.MNIST(root=path, train=False, transform=transform)
        # print(mnist_test.head())
        # 如果设置了max_data，则限制数据集的大小
        if max_data is not None:
            mnist_train = Subset(mnist_train, range(min(max_data, len(mnist_train))))
            mnist_test = Subset(mnist_test, range(min(max_data, len(mnist_test))))
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


def get_lids_random_batch(batchs, k=20):
    lids = []

    for X_batch in batchs:
        X_batch = X_batch.cpu().detach().numpy()
        lid_batch = mle_batch_np(X_batch, X_batch, k=k)

        lids.extend(lid_batch)

    lids = np.asarray(lids, dtype=np.float32)
    return np.mean(lids), np.std(lids)


def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    logits_list = []
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, (inputs, targets) in progress_bar:
        # print('start')
        # print(inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, logits = model(inputs)
        logits_list.append(logits)
        loss = criterion(outputs, targets)
        # print(f"{batch_idx}: loss:", loss.item())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # 更新进度条显示的信息
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
        # print('end')
    print('\n')
    train_loss = running_loss / len(data_loader)
    train_accuracy = 100. * correct / total
    lid_mean, lid_std = get_lids_random_batch(logits_list)
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
    lid_mean, lid_std = get_lids_random_batch(logits_list)
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
    # mlflow.set_tracking_uri("http://localhost:5002")
    mlflow.set_experiment("LID")
    mlflow.set_tracking_uri('file:./mlruns')
    # 获取数据集
    train_loader, test_loader = load_data(path='D:/gkw/data/classification', dataset_name=args.dataset,
                                          max_data=args.max_data, batch_size=args.batch_size,
                                          noise_ratio=args.noise_ratio, noise_type=args.noise_type)
    # if torch.cuda.is_available():
    #     train_loader = train_loader.cuda()
    #     test_loader = test_loader.cuda()
    # print("train_loader:", train_loader)
    # print("test_loader:", test_loader)
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # 设置模型
    if args.model == 'resnet18':
        model = ResNet18FeatureExtractor(pretrained=False, num_classes=args.num_classes)
        if torch.cuda.is_available():
            model = model.cuda()
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
    parser.add_argument('--max_data', default=256, type=int, help='max number of data')
    parser.add_argument('--noise_ratio', default=0.5, type=float, help='corruption ratio, should be less than 1')
    parser.add_argument('--noise_type', default='sym', type=str, help='[sym/asym]')
    parser.add_argument('--batch_size', default=15, type=int, help='batch size')
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
