import numpy as np
# from keras import backend as K
# import tensorflow as tf

import torch
import torch.nn.functional as F
from torch import nn


def symmetric_cross_entropy(alpha, beta):
    """
    Symmetric Cross Entropy:
    ICCV2019 "Symmetric Cross Entropy for Robust Learning with Noisy Labels"
    https://arxiv.org/abs/1908.06112
    """

    def loss(y_true, y_pred):
        # Ensure the predictions sum to 1 and are clamped to prevent log(0)
        y_pred = F.softmax(y_pred, dim=1)
        y_pred = torch.clamp(y_pred, min=1e-7, max=1.0)

        # Ensure the labels are in a valid range
        y_true = torch.clamp(y_true, min=1e-4, max=1.0)

        # Calculate the standard cross entropy (CE)
        ce = -torch.sum(y_true * torch.log(y_pred), dim=1)

        # Calculate the reverse cross entropy (RCE)
        rce = -torch.sum(y_pred * torch.log(y_true), dim=1)

        # Calculate the final symmetric cross entropy loss
        loss = alpha * ce.mean() + beta * rce.mean()
        return loss

    return loss


# def cross_entropy(y_true, y_pred):
#     return K.categorical_crossentropy(y_true, y_pred)


# def boot_soft(y_true, y_pred):
#     """
#     2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
#     https://arxiv.org/abs/1412.6596
#
#     :param y_true:
#     :param y_pred:
#     :return:
#     """
#     beta = 0.95
#
#     y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#     y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#     return -K.sum((beta * y_true + (1. - beta) * y_pred) *
#                   K.log(y_pred), axis=-1)


# def boot_hard(y_true, y_pred):
#     """
#     2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
#     https://arxiv.org/abs/1412.6596
#
#     :param y_true:
#     :param y_pred:
#     :return:
#     """
#     beta = 0.8
#
#     y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#     y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#     pred_labels = K.one_hot(K.argmax(y_pred, 1), num_classes=K.shape(y_true)[1])
#     return -K.sum((beta * y_true + (1. - beta) * pred_labels) *
#                   K.log(y_pred), axis=-1)


# def forward(P):
#     """
#     Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
#     CVPR17 https://arxiv.org/abs/1609.03683
#     :param P: noise model, a noisy label transition probability matrix
#     :return:
#     """
#     P = K.constant(P)
#
#     def loss(y_true, y_pred):
#         y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#         y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#         return -K.sum(y_true * K.log(K.dot(y_pred, P)), axis=-1)
#
#     return loss


# def backward(P):
#     """
#     Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
#     CVPR17 https://arxiv.org/abs/1609.03683
#     :param P: noise model, a noisy label transition probability matrix
#     :return:
#     """
#     P_inv = K.constant(np.linalg.inv(P))
#
#     def loss(y_true, y_pred):
#         y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#         y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#         return -K.sum(K.dot(y_true, P_inv) * K.log(y_pred), axis=-1)
#
#     return loss


# def lid(logits, k=20):
#     """
#     Calculate LID for each data point in the array.
#
#     :param logits:
#     :param k:
#     :return:
#     """
#     batch_size = tf.shape(logits)[0]
#     # n_samples = logits.get_shape().as_list()
#     # calculate pairwise distance
#     r = tf.reduce_sum(logits * logits, 1)
#     # turn r into column vector
#     r1 = tf.reshape(r, [-1, 1])
#     D = r1 - 2 * tf.matmul(logits, tf.transpose(logits)) + tf.transpose(r1) + \
#         tf.ones([batch_size, batch_size])
#
#     # find the k nearest neighbor
#     D1 = -tf.sqrt(D)
#     D2, _ = tf.nn.top_k(D1, k=k, sorted=True)
#     D3 = -D2[:, 1:]  # skip the x-to-x distance 0 by using [,1:]
#
#     m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
#     v_log = tf.reduce_sum(tf.log(m + K.epsilon()), axis=1)  # to avoid nan
#     lids = -k / v_log
#
#     return lids


# def lid_paced_loss(alpha=1.0, beta1=0.1, beta2=1.0):
#     """TO_DO
#     Class wise lid pace learning, targeting classwise asymetric label noise.
#
#     Args:
#       alpha: lid based adjustment paramter: this needs real-time update.
#     Returns:
#       Loss tensor of type float.
#     """
#     if alpha == 1.0:
#         return symmetric_cross_entropy(alpha=beta1, beta=beta2)
#     else:
#         def loss(y_true, y_pred):
#             pred_labels = K.one_hot(K.argmax(y_pred, 1), num_classes=K.shape(y_true)[1])
#             y_new = alpha * y_true + (1. - alpha) * pred_labels
#             y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#             y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#             return -K.sum(y_new * K.log(y_pred), axis=-1)
#
#         return loss

def lid(logits, k=20, device='cpu'):
    """
    使用PyTorch计算数组中每个数据点的局部内在维数（LID）。

    :param logits: 输入的logits张量
    :param k: 最近邻居的数量
    :param device: 设备类型，可以是'cpu'或'cuda'
    :return: 每个数据点的LID值
    """
    logits = logits.to(device)
    batch_size = logits.size(0)

    # 计算成对距离
    r = torch.sum(logits * logits, dim=1, keepdim=True)
    D = r - 2 * torch.mm(logits, logits.t()) + r.t() + torch.ones([batch_size, batch_size], device=device)

    # 找到k个最近邻居，不考虑自己到自己的距离
    D = -torch.sqrt(D)
    D, _ = torch.topk(D, k=k + 1, dim=1, largest=True, sorted=True)  # k+1因为包括了点自身
    D = -D[:, 1:]  # 忽略每个点到自己的距离

    # 使用最大距离对D进行归一化
    D /= D[:, -1].unsqueeze(1)
    D = torch.log(D + 1e-15)  # 防止log(0)，添加一个小的常数
    lids = -k / torch.sum(D, dim=1)  # 计算LID值

    return lids


def lid_paced_loss(alpha=1.0, beta1=0.1, beta2=1.0):
    """TO_DO
    Class wise lid pace learning, targeting classwise asymmetric label noise.

    Args:
      alpha: lid based adjustment parameter: this needs real-time update.
    Returns:
      Loss function compatible with PyTorch.
    """
    if alpha == 1.0:
        return symmetric_cross_entropy(alpha=beta1, beta=beta2)
    else:
        def loss(y_true, y_pred):
            num_classes = y_true.size(1)
            pred_labels = F.one_hot(torch.argmax(y_pred, dim=1), num_classes=num_classes).float()
            y_new = alpha * y_true + (1. - alpha) * pred_labels
            y_pred = F.softmax(y_pred, dim=-1)
            y_pred = torch.clamp(y_pred, min=torch.finfo(y_pred.dtype).eps, max=1. - torch.finfo(y_pred.dtype).eps)
            return -torch.sum(y_new * torch.log(y_pred), dim=-1)

        return loss
