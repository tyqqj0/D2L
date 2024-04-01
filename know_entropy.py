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
def inner_product_matrix(feature_maps, method='cosine'):
    """
    计算数据特征图内积矩阵
    :param feature_maps: 特征图列表 (group_size, C, H, W) torch
    :param method: 相似度计算方法
    :return: 内积矩阵
    """
    _, C, H, W = feature_maps.size()
    matrix = torch.zeros((C, C), device=feature_maps.device)

    # 将所有的H,W归一化
    if method == 'cosine':
        feature_maps = torch.nn.functional.normalize(feature_maps, p=2, dim=(2, 3))

    # 计算内积矩阵, 遍历h, w计算对应的gram求和
    for i in range(H):
        for j in range(W):
            # 选取特征图的一个像素点
            pixel = feature_maps[:, :, i, j]
            # 计算内积
            matrix += torch.mm(pixel.t(), pixel)

    return matrix


# 计算数据特征知识
def compute_knowledge(feature_maps, method='cosine'):
    """
        计算一组数据特征知识
        :param feature_maps: 特征图列表 (group_size, C, H, W)
        :param method: 相似度计算方法
        :return: 特征值矩阵， 特征向量矩阵(每个特征向量: (C, H, W))
        """
    # 如果是torch.Tensor类型，则复制
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.clone().detach()
    else:
        raise ValueError('Invalid type of feature_maps.')

    # 获取内积矩阵
    # n = len(feature_maps)
    matrix = inner_product_matrix(feature_maps, method)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    print(feature_maps.shape, matrix.shape)

    # 由于数值问题，特征值可能包含微小的负数，这里将它们置为零
    eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)

    # 归一化特征值，以避免除以零
    total = np.sum(eigenvalues)
    if total > 0:
        eigenvalues /= total
    else:
        # 如果所有特征值都是零，这意味着熵为零
        return 0

    # 对特征值排序
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]  # (C, C)， 其中每一列是一个特征向量, 应使用按列的方式索引即eigenvectors[:, i]

    # 计算原矩阵的特征图向量, 求group_size的特征图的平均特征图
    feature = np.zeros(feature_maps[0].shape)
    # print(feature.shape)
    for i in range(feature_maps.shape[0]):
        feature += feature_maps[i]
    feature /= feature_maps.shape[0]  # (C, H, W)
    # print(feature.shape)

    # 计算特征图向量，通过按照特征向量对特征图进行变换
    # feature_vector_matrices = np.einsum('ij,jklm->iklm', eigenvectors, feature_maps)
    vector_v = np.zeros((feature.shape[0], feature.shape[0], feature.shape[1], feature.shape[2]))  # (C, C, H, W)
    print(eigenvectors.shape, feature.shape)
    for i in range(len(eigenvectors)):
        for j in range(feature.shape[0]):
            a = eigenvectors[j][i] * feature[j]
            vector_v[j][i] = a

    # 交换第一第二维度方便索引值
    vector_v = np.swapaxes(vector_v, 0, 1)  # (C, C, H, W)

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
    # 获取特征值
    eigenvalues, _ = compute_knowledge(feature_maps, method)

    # 计算熵，忽略零特征值，因为 0 * log2(0) 应该是 0
    entropy = -np.sum(eigenvalues[eigenvalues > 0] * np.log2(eigenvalues[eigenvalues > 0]))

    return entropy
