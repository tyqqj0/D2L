# -*- CODING: UTF-8 -*-
# @time 2024/4/8 10:40
# @Author tyqqj
# @File MOF.py
# @
# @Aim 

import numpy as np
import torch


# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision


# 从两个类别的主成分中计算要修正的成分


# # 计算主成分2和主成分1中最大主成分的相关系数
# def compute_pca_corrcl(pca1, pca2, fimttt):
#     '''
#     计算两个主成分的相关系数矩阵
#     :param pca1: 主成分1 (C, C, H, W) (取0, C, H, W)
#     :param pca2: 主成分2 (C, C, H, W)
#     '''
#
#     assert pca1.shape == pca2.shape, 'The shape of pca1 and pca2 must be the same.'
#
#     # 计算相关系数矩阵
#     corr_matrix = np.zeros(pca2.shape[0])
#     i = 0
#     for j in range(i, pca2.shape[0]):
#         corr_matrix[i, j] = compute_vec_corr(pca1[i], pca2[j])
#         corr_matrix[j, i] = corr_matrix[i, j]
#
#     return corr_matrix
#
#
# # 计算两个主成分的相关系数矩阵
# def compute_pca_corr(pca1, pca2):
#     '''
#     计算两个主成分的相关系数矩阵
#     :param pca1: 主成分1 (C, C, H, W)
#     :param pca2: 主成分2 (C, C, H, W)
#     '''
#
#     assert pca1.shape == pca2.shape, 'The shape of pca1 and pca2 must be the same.'
#
#     # 计算相关系数矩阵
#     corr_matrix = np.zeros((pca1.shape[0], pca2.shape[0]))
#     for i in range(pca1.shape[0]):
#         for j in range(i, pca2.shape[0]):
#             corr_matrix[i, j] = compute_vec_corr(pca1[i], pca2[j])
#             corr_matrix[j, i] = corr_matrix[i, j]
#
#     return corr_matrix


def mof(x, label, num=10):
    if len(x.shape) == 4:
        x = x.view(x.shape[0], -1)
    # x(n, M), label(n)
    # 标准化并求协方差矩阵
    x = x - x.mean(dim=0)
    # 求x*xt获得数据关系矩阵
    xxt = torch.mm(x, x.t())
    # 将label不一致的数据的关系矩阵置零
    mask = label.unsqueeze(0) == label.unsqueeze(1)
    xxt = xxt * mask.float()

    # 求特征协方差矩阵 xtx = x-1 * xxt * x
    x_inv = torch.pinverse(x)
    xtx = torch.mm(x_inv, xxt)
    xtx = torch.mm(xtx, x)

    # 求特征值和特征向量
    eig_val, eig_vec = torch.linalg.eigh(xtx)

    # 按照大小排序
    # 按照大小排序
    idx = eig_val.argsort(descending=True)
    idx = idx[:num]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    # 标准化获得方向量
    norm = torch.norm(eig_vec, dim=0)

    # 处理norm为0的情况，若norm为0则将特征向量置为0
    eig_vec = eig_vec / norm
    eig_vec = eig_vec.where(norm > 0, torch.zeros_like(eig_vec))

    # 特征值归一化
    eig_val = eig_val / eig_val.sum()

    # 将eig_vec转置，方便索引数据
    eig_vec = eig_vec.t()

    return eig_vec, eig_val


def bkc(vec_allt, val_allt, all_classt, threshold=0.45):
    # vec_allt(num, M), all_classt(n)
    # 对主要方向进行清理，去除不同类别的与其他类别过度相似的次要方向

    # vec_allt: 所有类别的主要聚类块的方向,字典,键为类别标签,值为向量列表
    # val_allt: 所有类别的主要聚类块的特征值,字典,键为类别标签,值为特征值列表
    # all_classt: 所有类别的标签列表
    # threshold: 余弦相似度的阈值,超过这个阈值的向量对会被处理

    for i in range(len(all_classt)):
        for j in range(i + 1, len(all_classt)):
            # 获取两个类别的向量和特征值
            vec_i, val_i = vec_allt[all_classt[i]], val_allt[all_classt[i]]
            vec_j, val_j = vec_allt[all_classt[j]], val_allt[all_classt[j]]

            # 计算两个类别的每对向量的余弦相似度
            # print(vec_i.shape, vec_j.shape) # (10, 2048)
            eps = 1e-8  # 一个小的常数
            sim_matrix = torch.mm(vec_i, vec_j.t())
            sim_matrix = sim_matrix / torch.clamp(torch.norm(vec_i, dim=1).unsqueeze(1), min=eps)
            sim_matrix = sim_matrix / torch.clamp(torch.norm(vec_j, dim=1).unsqueeze(0), min=eps)
            sim_matrix = abs(sim_matrix)

            # print(sim_matrix)

            print(np.array2string(sim_matrix.cpu().numpy(), formatter={'float_kind': lambda x: "%.2f" % x}))

            # 找到余弦相似度超过阈值的向量对
            # indices = torch.where(sim_matrix > threshold) #abs()

            # 对每个超过阈值的向量对,去除特征值较小的向量
            for ii in range(vec_i.shape[0]):
                for jj in range(vec_j.shape[0]):
                    # 如果余弦相似度超过阈值,将特征值较小的特征方向置为零向量
                    if ii == 0 and jj == 0:
                        continue
                    if sim_matrix[ii, jj] > threshold:
                        print(
                            f'Similarity between class {all_classt[i]} direction {ii} and class {all_classt[j]} direction {jj}: {sim_matrix[ii, jj]}')
                        if val_i[ii] < val_j[jj]:
                            vec_i[ii] = torch.zeros_like(vec_i[ii])
                        else:
                            vec_j[jj] = torch.zeros_like(vec_j[jj])

            # 更新字典中的向量
            vec_allt[all_classt[i]] = vec_i
            vec_allt[all_classt[j]] = vec_j

    # 计算每个类还剩多少成分不为零
    num_components = {key: (torch.sum(vec_allt[key] != 0, dim=1) != 0).sum().item() for key in all_classt}
    print(num_components)

    return vec_allt


def compute_pca_correct(pca1, pca2, fimttt):
    '''
    从两个类别的主成分中计算要修正的成分
    :param pca1: 类别1的主成分 (C, C, H, W)
    :param pca2: 类别2的主成分 (C, C, H, W)
    '''
    assert pca1.shape == pca2.shape, 'The shape of pca1 and pca2 must be the same.'
    # 如果特征图大小为一，则计算2中与1最相关的主成分
    if len(pca1.shape) == 2:
        # print(pca1[0])
        pca1 = pca1[0].clone()  # C
        pca2 = pca2.clone()  # (C, C)

        # 标准化
        pca1 = pca1 / torch.norm(pca1)
        pca2 = pca2 / torch.norm(pca2, dim=1, keepdim=True).where(torch.norm(pca2, dim=1, keepdim=True) > 0,
                                                                  torch.ones_like(pca2))

        # 标准化
        pca1 = pca1 / torch.norm(pca1)
        pca2 = pca2 / torch.norm(pca2, dim=1, keepdim=True).where(torch.norm(pca2, dim=1, keepdim=True) > 0,
                                                                  torch.ones_like(pca2))

        # 计算相关系数
        corr_matrix = torch.mm(pca2, pca1.view(pca1.size(0), -1))  # (C)

        # 找到最大相关系数
        index1 = 0
        indexmax = torch.argmax(corr_matrix[1:])

        # 如果最大相关系数小于主要相关系数，confdt会小于0
        if abs(corr_matrix[index1]) > abs(corr_matrix[indexmax]):
            confdt = 0
        else:
            # 计算偏离置信程度
            confdt = 1 - (abs(corr_matrix[index1]) / abs(corr_matrix[indexmax]))
            if confdt < 0:
                confdt = 0
            else:
                confdt = confdt.item()

        # 打印信息检查
        # print('main_cor:{}, max_cor:{}'.format(corr_matrix[index1].item(), corr_matrix[indexmax].item()))
        # print('confdt', confdt.item())

        # 取出最大相关系数对应的主成分
        pca_correct2 = pca2[indexmax]

        return pca_correct2, confdt, corr_matrix[index1]

    # 复制值
    pca1 = pca1.clone()
    pca2 = pca2.clone()

    shape = pca1.shape
    # 取出类别1中最大主成分
    pca1 = pca1[0]  # (C, H, W)

    # 整理
    pca1 = pca1.view(-1)  # (C*H*W)
    pca2 = pca2.view(pca2.shape[0], -1)  # (C, C*H*W)

    # 计算范数
    norm_pca1 = torch.norm(pca1)
    norm_pca2 = torch.norm(pca2, dim=1, keepdim=True)

    # 避免除以零的情况
    pca1 = pca1 / norm_pca1 if norm_pca1 > 0 else pca1
    pca2 = pca2 / norm_pca2.where(norm_pca2 > 0, torch.ones_like(norm_pca2))

    # 计算要修正的成分, 找到主成分2与主成分1最大成分最相关的主成分
    corr_index = torch.matmul(pca2, pca1)  # (C)

    index1 = 0  # 两个类别最大主成分的相关系数
    indexmax = torch.argmax(corr_index[1:])  # 最大相关系数

    # 计算偏离置信程度
    confdt = 1 - (corr_index[indexmax] / corr_index[index1])

    # 找到最大的相关系数
    # 取出最大相关系数对应的主成分
    pca_correct2 = pca2[indexmax]  # (C*H*W)

    # 还原形状
    pca_correct2 = pca_correct2.view(shape[1], shape[2], shape[3])  # (C, H, W)
    # print(pca_correct2.shape)

    return pca_correct2, confdt, corr_index[indexmax]


def compute_vec_corr(vec1, vec2):
    '''
    计算两个向量的相关系数
    :param vec1: 向量1 (C, H, W)
    :param vec2: 向量2 (C, H, W)
    :return: 相关系数
    '''
    # 复制值
    vec1 = vec1.clone()
    vec2 = vec2.clone()
    assert vec1.shape == vec2.shape, 'The shape of vec1 and vec2 must be the same. {}, {}'.format(vec1.shape,
                                                                                                  vec2.shape)
    # print(vec1[1], vec2[1])

    # 如果传入的是向量C
    if len(vec1.shape) == 1:
        # 计算相关系数
        inner_product = torch.dot(vec1, vec2)
        norm1 = torch.norm(vec1)
        norm2 = torch.norm(vec2)

        # 避免除以零的情况
        norm1 = norm1 if norm1 > 0 else 1
        norm2 = norm2 if norm2 > 0 else 1

        corr = inner_product / (norm1 * norm2)

        return corr.item()

    # 获取特征图的大小
    _, H, W = vec1.size()

    # 如果 H 和 W 大于1，那么执行去均值操作
    if H > 1 and W > 1:
        vec1_mean = vec1.mean(dim=(1, 2), keepdim=True)
        vec2_mean = vec2.mean(dim=(1, 2), keepdim=True)

        vec1 = vec1 - vec1_mean
        vec2 = vec2 - vec2_mean

    # print(vec1, vec2)

    # 计算标准化后的向量的内积
    inner_product = (vec1 * vec2).sum(dim=(1, 2))

    # 计算向量的模
    norm1 = torch.sqrt((vec1 * vec1).sum(dim=(1, 2)))
    norm2 = torch.sqrt((vec2 * vec2).sum(dim=(1, 2)))

    # 避免除以零的情况
    valid = (norm1 != 0) & (norm2 != 0)
    corr = torch.where(valid, inner_product / (norm1 * norm2), torch.zeros_like(inner_product))

    # Additional debugging information
    # print(f"Inner products: {inner_product}")
    # print(f"Norms of vec1: {norm1}")
    # print(f"Norms of vec2: {norm2}")
    # print(f"Valid mask: {valid}")
    # print(f"Correlation coefficients: {corr}")

    # 计算相关系数的累加值
    corr_sum = abs(corr).sum()

    # 然后除以元素的总数来得到平均值
    corr_mean = corr_sum / corr.numel()

    return corr_mean.item()  # 返回标量值
