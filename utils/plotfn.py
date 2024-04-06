# -*- CODING: UTF-8 -*-
# @time 2024/3/20 18:08
# @Author tyqqj
# @File plotfn.py
# @
# @Aim


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE


def plot_lid_plt(lidss, epoch, y_lim=None, folder='', pre='', path=None):
    file_name = folder + '/' + pre + '_' + 'epoch_{:03d}.png'.format(epoch)
    # 如果文件夹不存在
    if not os.path.exists(path + folder):
        os.makedirs(path + folder)
    import matplotlib.pyplot as plt
    plt.figure()
    layers = list(lidss.keys())
    values = [lidss[layer] for layer in layers]
    plt.bar(layers, values)
    if y_lim:
        plt.ylim((0, y_lim))
    plt.xlabel('Layers')
    plt.ylabel('Values')
    plt.title(f'{pre} Layer {folder} at Epoch {epoch}')
    # plt.legend()
    plt.savefig(path + file_name)
    print('save plot {}'.format(file_name))
    plt.close()
    return path


def plot_layers_seaborn(lidss, epoch, y_lim=None, folder='', pre='', path=None):
    file_name = pre + '_' + 'epoch_{:03d}.png'.format(epoch)
    # 如果path不为None，则在path中创建文件夹
    full_folder_path = os.path.join(path, folder) if path is not None else folder
    if not os.path.exists(full_folder_path):
        os.makedirs(full_folder_path)
    full_file_path = os.path.join(full_folder_path, file_name)

    # 将字典转换为DataFrame
    data = pd.DataFrame(list(lidss.items()), columns=['Layers', 'Values'])

    # 使用Seaborn设置风格
    sns.set_theme(style="whitegrid")

    # 使用Seaborn设置调色板
    sns.set_palette("pastel")

    # 创建Seaborn条形图
    plt.figure()
    barplot = sns.barplot(x='Layers', y='Values', data=data, errorbar=None)

    # 设置y轴的限制
    if y_lim:
        barplot.set(ylim=(0, y_lim))

    # 设置x轴和y轴的标签
    plt.xlabel('Layers')
    plt.ylabel('Values')

    # 设置图表的标题
    plt.title(f'{pre} Layer {folder} at Epoch {epoch}')

    # 保存图表
    plt.savefig(full_file_path)
    print('Saved plot {}'.format(file_name))

    # 关闭图表以释放内存
    plt.close()

    return full_folder_path


def kn_map(data_epoch, label, epoch, group_size=25, folder='', pre='', path=None):
    '''
    :param data_epoch: 二维数据, (batch_size, feature_dim)
    :param label_epoch: 一维数据, (batch_size, )
    :param epoch: int, epoch
    :param folder: str, 文件夹名称
    :param path: str, 路径
    :return: plot
    '''
    folder = folder + '/kn_map{:03d}'.format(epoch)
    file_name = pre + '_' + 'epoch_{:03d}'.format(epoch) + '_'
    # 如果path不为None，则在path中创建文件夹

    if not os.path.exists(path + folder):
        os.makedirs(path + folder)
    full_file_path = os.path.join(path, folder, file_name)

    # 运行t-SNE降维
    for layer, data in data_epoch.items():
        layer_plt = kn_map_layer(data, label, layer, group_size=group_size)
        layer_plt.savefig(full_file_path + layer + '.png')
        layer_plt.close()
        print('Saved plot {}'.format(layer + '.png'))

    # 跑一张整体图，带不同标签
    # data = np.vstack([data for data in data_epoch.values()])
    # layer_plt = kn_map_layer(data, label, 'all', group_size=group_size)

    return os.path.join(path, folder)


# 利用t-sne对各层输入数据降维可视化,使用seaborn绘图
def kn_map_layer(data, label, layer='', group_size=25):
    """
    :param data: 二维数据, (batch_size, feature_dim)
    :param label: 一维标签数组
    :param layer: str, 层名称
    :param group_size: t-SNE的perplexity参数将基于此值进行设置
    :return: plot
    """

    # 确保数据是二维的
    if len(data.shape) != 2:
        raise ValueError("Data should be 2D array.")

    # 将数据从PyTorch张量转换为NumPy数组
    data = data.cpu().detach().numpy()

    # # 确保标签是一维的
    # if len(label.shape) != 1:
    #     raise ValueError("Label should be 1D array.")

    # # 确保数据和标签的样本数目相同
    # if data.shape[0] != label.shape[0]:
    #     raise ValueError("Data and label must have the same number of samples.")

    # 运行t-SNE降维
    tsne = TSNE(n_components=2, perplexity=min(group_size * 2 / 3, data.shape[0] / 3), n_iter=1000)
    data_tsne = tsne.fit_transform(data)

    # 将降维后的数据和标签转换为DataFrame, 用label作为hue, 即颜色
    df = pd.DataFrame(data_tsne, columns=['Dim1', 'Dim2'])
    df['label'] = label

    # 使用Seaborn设置风格和调色板
    # 不画轴名称
    plt.axis('off')

    # 使用Seaborn设置风格
    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")

    # 创建Seaborn散点图
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(x='Dim1', y='Dim2', hue='label', data=df, palette="deep", legend='full')

    # 不显示坐标轴名称
    plt.axis('off')

    # 设置图表的标题
    plt.title(f'{layer} t-SNE')

    # 返回matplotlib的plt对象，调用者可以使用plt.show()来显示或保存图像
    return plt


label_of_cifar10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def scale_image(image):
    if image.dtype.kind == 'f':
        # 归一化到 [0, 1]
        if image.min() < 0 or image.max() > 1:
            image = (image - image.min()) / (image.max() - image.min())

    # 如果图像数据是整数
    elif image.dtype.kind == 'u' or image.dtype.kind == 'i':
        # 确保数据在 [0, 255] 范围内
        image = np.clip(image, 0, 255)

    return image


# 可视化错误标签
def plot_wrong_label(data, label, pred, epoch, folder='', pre='', path=None, max_samples=10, replace_label=False):
    '''
    :param data: 二维数据, (batch_size, feature_dim)
    :param label: 一维数据, (batch_size, )
    :param pred: 一维数据, (batch_size, )
    :param epoch: int, epoch
    :param folder: str, 文件夹名称
    :param path: str, 路径
    :return: plot
    '''
    folder = folder
    file_name = pre + '_' + 'epoch_{:03d}'.format(epoch)
    # 如果path不为None，则在path中创建文件夹

    if not os.path.exists(path + folder):
        os.makedirs(path + folder)
    full_file_path = os.path.join(path, folder, file_name)
    label = label.cpu().numpy()
    pred = pred.cpu().numpy()
    data = data.cpu().numpy()

    # 如果label是one-hot编码，则转换为标量
    if len(label.shape) > 1:
        label = np.argmax(label, axis=1)

    if len(pred.shape) > 1:
        pred = np.argmax(pred, axis=1)

    # 获取错误标签的索引
    wrong_idx = np.where(label != pred)[0]

    # 如果错误标签的数量超过最大样本数，则随机选择最大样本数个错误标签
    if len(wrong_idx) > max_samples:
        wrong_idx = np.random.choice(wrong_idx, max_samples, replace=False)

    if len(wrong_idx) == 0:
        return None

    # 选取错误标签的数据和标签
    # print(data.shape)
    data = np.transpose(data[wrong_idx], (0, 2, 3, 1))
    label = label[wrong_idx]
    pred = pred[wrong_idx]

    # 创建Seaborn图
    plt.figure()
    # print('replace label:', replace_label)
    for i in range(len(data)):
        plt.subplot(1, 6, i + 1)
        plt.imshow(scale_image(data[i]), cmap='gray', interpolation='none')
        plt.title(
            f'True: {label_of_cifar10[label[i]] if replace_label else label[i]}\n{label_of_cifar10[pred[i]] if replace_label else label[i]}')
        plt.axis('off')

    # 保存图表
    plt.savefig(full_file_path + '.png')

    plt.close()

    return os.path.join(path, folder)





# 绘制一些图像
def plot_images(images, epoch, folder='', pre='', path=None, max_samples=3, replace_label=False):
    '''
    :param images: 二维数据, (batch_size, feature_dim)
    :param labels: 一维数据, (batch_size, )
    :param epoch: int, epoch
    :param folder: str, 文件夹名称
    :param path: str, 路径
    :return: plot
    '''
    if len(images) == 0:
        return None
    folder = folder
    file_name = pre + '_' + 'epoch_{:03d}'.format(epoch)
    # 如果path不为None，则在path中创建文件夹

    if not os.path.exists(path + folder):
        os.makedirs(path + folder)
    full_file_path = os.path.join(path, folder, file_name)
    # 转换 images 到 NumPy 数组
    if isinstance(images, list):
        # 如果是列表，确保所有元素形状一致
        if all(image.shape == images[0].shape for image in images):
            # 可以安全地转换为 NumPy 数组
            images = np.stack(images)
        else:
            raise ValueError("All elements in the images list must have the same shape.")
    elif isinstance(images, torch.Tensor):
        # 如果是 PyTorch 张量，直接转换为 NumPy 数组
        images = images.numpy()
    if images.ndim == 4:  # 如果 images 是四维张量 (batch_size, C, H, W)
        images = images.transpose((0, 2, 3, 1))



    # 选取前max_samples个图像
    idx = np.random.choice(len(images), max_samples, replace=False)
    # print(data.shape)
    images = np.transpose(images, (0, 2, 3, 1))[idx]



    # 创建Seaborn图
    plt.figure()
    # print('replace label:', replace_label)
    for i in range(len(images)):
        plt.subplot(1, 6, i + 1)
        plt.imshow(scale_image(images[i].squeeze()), cmap='gray', interpolation='none')
        # plt.title(
        #     f'{label_of_cifar10[labels[i]] if replace_label else labels[i]}')
        plt.axis('off')

    # 保存图表
    plt.savefig(full_file_path + '.png')

    plt.close()

    return os.path.join(path, folder)


# 将knowledge的字典转换为json并保存提交

def dict_to_json(dicttt, epoch, folder='knowledge_json', pre='', path=''):
    import json
    import os
    file_name = pre + '_' + 'epoch_{:03d}.json'.format(epoch)
    # 如果path不为None，则在path中创建文件夹
    full_folder_path = os.path.join(path, folder) if path is not None else folder
    if not os.path.exists(full_folder_path):
        os.makedirs(full_folder_path)
    full_file_path = full_folder_path + '/'
    full_file_path = full_file_path + file_name

    with open(full_file_path, 'w') as f:
        # 处理格式为自动缩进

        json.dump(dicttt, f)
    return full_folder_path
