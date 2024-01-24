# -*- CODING: UTF-8 -*-
# @time 2024/1/24 19:19
# @Author tyqqj
# @File LID.py
# @
# @Aim
import numpy as np
from scipy.spatial.distance import cdist
compute_dist = cdist


def mle_batch_np(data, batch, k):
    """
    lid of a batch of query points X.
    numpy implementation.
    LID(X, Xs) = - k / sum_{i=1}^{k} log (d(x, x_i) / d(x, x_k+1)

    :param data:
    :param batch:
    :param k:
    :return:
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1] + 1e-8))
    a = compute_dist(batch, data)  # 计算两个矩阵的距离，返回一个矩阵
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]  # 按行排序
    a = np.apply_along_axis(f, axis=1, arr=a)  # 按行计算，
    return a
