# -*- CODING: UTF-8 -*-
# @time 2024/1/25 12:01
# @Author tyqqj
# @File data.py
# @
# @Aim
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms, datasets


class NoiseTransitionMatrix:
    def __init__(self, num_classes, seed=None):
        self.num_classes = num_classes
        self.seed = seed
        self.matrix = create_noise_transition_matrix(num_classes, seed)

    def __str__(self):
        return self.matrix.__str__()

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if row >= self.num_classes or col >= self.num_classes:
                raise IndexError("Row or column index out of bounds.")
            return self.matrix[row, col]
        else:
            raise TypeError("Invalid key type. Key must be a tuple of two integers.")

    def __mul__(self, other):
        # 只接受标量乘，标量范围在01之间，对角线不作乘法而是保持行之和为1
        if isinstance(other, (int, float)):
            if other < 0 or other > 1:
                raise ValueError("Scalar value must be between 0 and 1.")
            # 去除原始矩阵的对角线
            matrix = self.matrix - np.diag(np.diag(self.matrix))
            # 乘以标量
            matrix = matrix * other
            # 求出行之和，若不到一，则将剩余部分加到对角线上
            row_sums = matrix.sum(axis=1)
            for i in range(self.num_classes):
                if row_sums[i] < 1:
                    matrix[i, i] += 1 - row_sums[i]
            # 行归一化到1
            matrix = matrix / matrix.sum(axis=1)[:, np.newaxis]
            return matrix

    def get(self):
        return self.matrix


class NoisyDataset(Dataset):
    def __init__(self, dataset, noise_ratio=0.0, noise_type='sym'):
        self.origin_dataset = dataset
        self.noise_ratio = noise_ratio
        self.noise_type = noise_type
        if noise_type not in ['sym', 'asym']:
            raise ValueError("Invalid noise type. Must be one of ['sym', 'asym'].")
        if noise_type == 'asym' and noise_ratio > 0:
            self.noise_matrix = NoiseTransitionMatrix(dataset.num_classes)
        # self.origin_dataset = None
        # 检查是否是Subset，如果是，则直接使用原始数据集的targets属性
        if isinstance(self.origin_dataset, Subset):
            self.targets = self.origin_dataset.dataset.targets
        else:
            self.targets = self.origin_dataset.targets

        # 计算类别数量
        self.num_classes = len(np.unique(self.targets))

        self.apply_noise()

    def apply_noise(self):
        if self.noise_ratio > 0:
            n_samples = len(self.origin_dataset)
            n_noisy = int(self.noise_ratio * n_samples)
            indices = np.random.choice(n_samples, n_noisy, replace=False)
            for idx in indices:
                if self.noise_type == 'sym':
                    # 生成新的随机标签
                    new_label = torch.randint(0, self.num_classes, (1,)).item()
                    # 确保新标签与原标签不同
                    while new_label == self.targets[idx]:
                        new_label = torch.randint(0, self.num_classes, (1,)).item()
                    # 应用噪声
                    self.targets[idx] = new_label
                elif self.noise_type == 'asym':
                    # 利用噪声转移矩阵做标签转换
                    self.targets[idx] = torch.tensor(
                        np.random.choice(self.num_classes, 1, p=self.noise_matrix[self.targets[idx]]))

    def __len__(self):
        return len(self.origin_dataset)

    def __getitem__(self, idx):
        if isinstance(self.origin_dataset, Subset):
            # 如果是Subset，则从原始数据集中获取数据和标签
            img, _ = self.origin_dataset.dataset[self.origin_dataset.indices[idx]]
            target = self.targets[self.origin_dataset.indices[idx]]
        else:
            # 如果不是Subset，则直接从数据集中获取数据和标签
            img, target = self.origin_dataset[idx]

        return img, target


def load_data(path='D:/gkw/data/classification', max_data=1024, dataset_name='MNIST', batch_size=128, noise_ratio=0.0,
              noise_type='sym'):
    num_classes = -1

    # if dataset_name == 'MNIST':
    num_classes = -1
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    if dataset_name == 'MNIST':
        num_classes = 10
        transform = transform_mnist
    elif dataset_name == 'CIFAR10':
        num_classes = 10
        transform = transform_cifar

    # 通用的数据集加载逻辑
    train_dataset = datasets.__dict__[dataset_name](root=path, train=True, transform=transform, download=True)
    test_dataset = datasets.__dict__[dataset_name](root=path, train=False, transform=transform)

    # 如果设置了max_data，则限制数据集的大小
    if max_data is not None:
        train_dataset = Subset(train_dataset, range(min(max_data, len(train_dataset))))
        test_dataset = Subset(test_dataset, range(min(max_data, len(test_dataset))))

    # 应用噪声
    train_dataset = NoisyDataset(train_dataset, noise_ratio=noise_ratio,
                                 noise_type=noise_type) if noise_ratio > 0 else train_dataset
    test_dataset = test_dataset  # 测试集通常不添加噪声

    # 数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, num_classes


def create_noise_transition_matrix(num_classes, seed=None):
    """
    创建一个噪声转移矩阵。

    参数:
    num_classes (int): 类别的数量。
    seed (int): 随机数生成器的种子。

    返回:
    np.array: 噪声转移矩阵。
    """

    # 固定随机种子以确保可重现性
    if seed is not None:
        np.random.seed(seed)

    # 创建一个num_classes x num_classes的矩阵，每个元素是[0, 1]内的随机数
    noise_matrix = np.random.rand(num_classes, num_classes)

    # 使每一行加起来等于1
    noise_matrix_normalized = noise_matrix / noise_matrix.sum(axis=1)[:, np.newaxis]

    return noise_matrix_normalized
