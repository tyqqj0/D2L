# -*- CODING: UTF-8 -*-
# @time 2024/3/26 20:27
# @Author tyqqj
# @File know_entropy.py
# @
# @Aim 


import numpy as np
import sympy as sp
import os
import matplotlib.pyplot as plt
import torch
import torch.distributions


# import cv2


# 两张特征图的相似度计算
def euclidean_distance_batch(feature_maps1, feature_maps2):
    # Assuming feature_maps1 and feature_maps2 are of shape (batch_size, *feature_dimensions)
    diff = feature_maps1 - feature_maps2
    distance = np.sqrt(np.sum(np.square(diff), axis=tuple(range(1, diff.ndim))))
    return distance


def cosine_similarity_batch(feature_maps1, feature_maps2, epsilon=1e-10):
    # Flatten all but the first dimension for dot product calculation
    flattened_maps1 = feature_maps1.reshape(feature_maps1.shape[0], -1)
    flattened_maps2 = feature_maps2.reshape(feature_maps2.shape[0], -1)

    dot_product = np.sum(flattened_maps1 * flattened_maps2, axis=1)
    norm1 = np.linalg.norm(flattened_maps1, axis=1)
    norm2 = np.linalg.norm(flattened_maps2, axis=1)

    similarity = dot_product / (np.maximum(norm1 * norm2, epsilon))
    return similarity


def pearson_correlation_coefficient_batch(feature_maps1, feature_maps2):
    # Flatten all but the first dimension for mean and std calculations
    flattened_maps1 = feature_maps1.reshape(feature_maps1.shape[0], -1)
    flattened_maps2 = feature_maps2.reshape(feature_maps2.shape[0], -1)

    mean1 = np.mean(flattened_maps1, axis=1)
    mean2 = np.mean(flattened_maps2, axis=1)
    std1 = np.std(flattened_maps1, axis=1)
    std2 = np.std(flattened_maps2, axis=1)

    normalized_maps1 = (flattened_maps1 - mean1[:, np.newaxis]) / std1[:, np.newaxis]
    normalized_maps2 = (flattened_maps2 - mean2[:, np.newaxis]) / std2[:, np.newaxis]

    correlation = np.sum(normalized_maps1 * normalized_maps2, axis=1) / (flattened_maps1.shape[1] - 1)
    return correlation


class FeatureMapSimilarityBatch:
    def __init__(self, method='cosine'):
        self.methods = {
            'euclidean': euclidean_distance_batch,
            'cosine': cosine_similarity_batch,
            'pearson': pearson_correlation_coefficient_batch
        }
        if method not in self.methods:
            raise ValueError(f"Method not recognized. Available methods: {list(self.methods.keys())}")
        self.method = method

    def __call__(self, feature_maps1, feature_maps2):
        if feature_maps1.shape != feature_maps2.shape:
            raise ValueError("The two batches of feature maps must have the same shape.")
        # 如果 feature_maps1 和 feature_maps2 都是 1 维的，直接计算并返回它们的点积
        if feature_maps1.ndim == 1 and feature_maps2.ndim == 1:
            return np.dot(feature_maps1, feature_maps2)
        return self.methods[self.method](feature_maps1, feature_maps2)

    def set_method(self, method):
        if method not in self.methods:
            raise ValueError(f"Method not recognized. Available methods: {list(self.methods.keys())}")
        self.method = method


# Example usage:
# feature_maps1 and feature_maps2 are numpy arrays of shape (batch_size, *feature_map_dimensions)

# similarity_calculator = FeatureMapSimilarityBatch(method='cosine')
# similarities = similarity_calculator(feature_maps1, feature_maps2)
def euclidean_distance(feature_map1, feature_map2):
    """
    计算两个特征图的欧氏距离
    :param feature_map1: 特征图1
    :param feature_map2: 特征图2
    :return: 欧氏距离
    """
    # 计算两个特征图的差值
    diff = feature_map1 - feature_map2
    # 计算欧氏距离
    distance = np.sqrt(np.sum(diff ** 2))
    return distance


def cosine_similarity(feature_map1, feature_map2, epsilon=1e-5):
    """
    计算两个特征图的余弦相似度
    :param feature_map1: 特征图1
    :param feature_map2: 特征图2
    :return: 余弦相似度
    """
    # 计算两个特征图的内积
    dot_product = np.sum(feature_map1 * feature_map2)
    # if dot_product ==
    # 计算两个特征图的模
    norm1 = np.sqrt(np.sum(feature_map1 ** 2))
    norm2 = np.sqrt(np.sum(feature_map2 ** 2))
    # 计算余弦相似度
    similarity = dot_product / (max(norm1 * norm2, epsilon))
    # print(similarity)

    return similarity


def pearson_correlation_coefficient(feature_map1, feature_map2):
    """
    计算两个特征图的皮尔逊相关系数
    :param feature_map1: 特征图1
    :param feature_map2: 特征图2
    :return: 皮尔逊相关系数
    """
    # 计算两个特征图的均值
    mean1 = np.mean(feature_map1)
    mean2 = np.mean(feature_map2)
    # 计算两个特征图的差值
    diff1 = feature_map1 - mean1
    diff2 = feature_map2 - mean2
    # 计算皮


class FeatureMapSimilarity:
    def __init__(self, method='cosine'):
        self.method = method

    def __call__(self, feature_map1, feature_map2):
        feature_map1 = np.asarray(feature_map1, dtype=np.float32)
        feature_map2 = np.asarray(feature_map2, dtype=np.float32)
        similarity = self.similarity(feature_map1, feature_map2)
        if np.isinf(similarity):
            similarity = 1
        if similarity == 0 or np.isnan(similarity):
            similarity = 1e-9
        # print(similarity)
        return similarity

    def similarity(self, feature_map1, feature_map2):
        if self.method == 'cosine':
            return cosine_similarity(feature_map1, feature_map2)
        elif self.method == 'euclidean':
            return euclidean_distance(feature_map1, feature_map2)
        elif self.method == 'pearson':
            return pearson_correlation_coefficient(feature_map1, feature_map2)
        else:
            raise ValueError('Invalid method.')

    def __apply__(self, feature_map1, feature_map2):
        return self.similarity(feature_map1, feature_map2) + 1e-9


# 计算数据特征图内积矩阵
def inner_product_matrix(feature_maps, method=None):
    """
    计算数据特征图内积矩阵
    :param feature_maps: 特征图列表 (group_size, C, H, W) torch
    :param method: 相似度计算方法
    :return: 内积矩阵
    """
    if len(feature_maps.shape) == 2:
        return torch.matmul(feature_maps, feature_maps.t())
    n, C, H, W = feature_maps.size()
    matrix = torch.zeros((C, C), device=feature_maps.device)

    # 如果使用 'cosine' 方法，归一化特征图
    if method == 'cosine':
        feature_maps = torch.nn.functional.normalize(feature_maps, p=2, dim=(2, 3))

    # 计算内积矩阵, 重整特征图
    feature_maps = feature_maps.view(n, C, -1)  # (group_size, C, H*W)

    # 使用 PyTorch
    # 重整特征图以便进行批处理
    feature_maps = feature_maps.view(n, C, H * W)

    # 对批量数据进行内积计算。由于 feature_maps 已经是 (n, C, H*W) 的形状，
    # 我们可以直接使用 torch.matmul 进行批量计算
    # 结果 matrix 的形状将是 (n, C, C)
    matrix = torch.matmul(feature_maps, feature_maps.transpose(1, 2))

    # 将整个批次的结果求和，得到最终的内积矩阵
    matrix = matrix.sum(dim=0)

    # print(matrix.shape)
    return matrix


# 计算数据特征知识
def compute_knowledge(feature_maps, method='cosine', normalize=True):
    """
        计算一组数据特征知识
        :param feature_maps: 特征图列表 (group_size, C, H, W)
        :param method: 相似度计算方法
        :return: 特征值矩阵， 特征向量矩阵(每个特征向量: (C, H, W))
        """
    if normalize:
        feature_maps = torch.nn.functional.normalize(feature_maps, p=2, dim=(2, 3))
    if method == 'dot' or len(feature_maps.shape) == 2:
        # 将特征图张量转换为矩阵
        # print(feature_maps.shape[1])
        feature_maps1 = feature_maps.view(feature_maps.size(0), -1)
        # 计算内积矩阵
        matrix = torch.matmul(feature_maps1.t(), feature_maps1)
        # print(matrix.shape)
        eigenvalues, eigenvectors = torch.linalg.eig(matrix)
        # print(eigenvectors.shape)
        eigenvalues = eigenvalues.real  # 取实数部分
        eigenvectors = eigenvectors.real

        # 由于数值问题，特征值可能包含微小的负数，这里将它们置为零
        eigenvalues = torch.clamp(eigenvalues, min=0)

        # 归一化特征值，以避免除以零
        total = torch.sum(eigenvalues)
        if total > 0:
            eigenvalues /= total

        # 对特征值排序
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]  # (C, C)，每一列是一个特征向量

        return eigenvalues, eigenvectors

    else:
        # 如果是torch.Tensor类型，则复制
        # 获取内积矩阵
        matrix = inner_product_matrix(feature_maps, method)
    eigenvalues, eigenvectors = torch.linalg.eig(matrix)
    eigenvalues = eigenvalues.real  # 取实数部分
    eigenvectors = eigenvectors.real

    # 由于数值问题，特征值可能包含微小的负数，这里将它们置为零
    eigenvalues = torch.clamp(eigenvalues, min=0)

    # 归一化特征值，以避免除以零
    total = torch.sum(eigenvalues)
    if total > 0:
        eigenvalues /= total

    # 对特征值排序
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]  # (C, C)，每一列是一个特征向量

    # 计算原矩阵的特征图向量, 求group_size的特征图的平均特征图
    feature = torch.mean(feature_maps, dim=0)  # (C, H, W)
    # print(feature.shape, eigenvectors.shape)
    # 计算特征图向量，通过按照特征向量对特征图进行变换
    n, C, H, W = feature_maps.shape
    # print(feature.device)
    # print(feature.shape,eigenvectors.shape)
    # 创建一个形状为 [C, C, H, W] 的全零张量
    diag_feature = torch.zeros(C, C, H, W, device=feature.device)
    # print(diag_feature.shape)

    # 填充对角线
    for i in range(C):
        diag_feature[i, i] = feature[i]
    vector_v = torch.matmul(eigenvectors, diag_feature.view(C, C, -1))

    # 重塑 vector_v 以获得每个特征对应的空间布局
    # 每一个主成分都应该有形状 [H, W]
    vector_v = vector_v.view(C, C, H, W)  # 结果应该是 [64, 64, 14, 14]
    # print(vector_v.shape)
    vector_v = vector_v.view(eigenvectors.shape[1], *feature.shape)  # (C, C, H, W)

    # 返回特征值和特征向量
    return eigenvalues, vector_v


# 计算一组数据特征图的知识熵
def knowledge_entropy(feature_maps, method='cosine'):
    """
    计算一组数据特征图的知识熵
    :param feature_maps: 特征图列表 (group_size, C, H, W)
    :param method: 相似度计算方法
    :return: 知识熵
    """
    # 假设 compute_knowledge 是一个返回特征值的函数，并且这些特征值已经在 [0, 1] 范围内归一化
    eigenvalues, _ = compute_knowledge(feature_maps, method)

    # 选择大于0的特征值
    valid_eigenvalues = eigenvalues[eigenvalues > 0]

    # 计算熵，忽略零特征值
    # 使用 PyTorch 的 log 函数和乘法
    entropy = -torch.sum(valid_eigenvalues * torch.log2(valid_eigenvalues))

    # 将结果转换为标量，如果需要的话
    return entropy.item()
