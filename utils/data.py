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
