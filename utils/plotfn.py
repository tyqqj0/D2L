# -*- CODING: UTF-8 -*-
# @time 2024/3/20 18:08
# @Author tyqqj
# @File plotfn.py
# @
# @Aim


import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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


def plot_lid_seaborn(lidss, epoch, y_lim=None, folder='', pre='', path=None):
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



def kn_map(data_epoch, label, epoch, group_size, folder='', pre='', path=None):
    '''
    :param data_epoch: 二维数据, (batch_size, feature_dim)
    :param label_epoch: 一维数据, (batch_size, )
    :param epoch: int, epoch
    :param folder: str, 文件夹名称
    :param path: str, 路径
    :return: plot
    '''
    folder = folder + '/kn_map{:03d}_label{}'.format(epoch, label)
    file_name = pre + '_' + 'epoch_{:03d}'.format(epoch)
    # 如果path不为None，则在path中创建文件夹

    if not os.path.exists(path + folder):
        os.makedirs(path + folder)
    full_file_path = os.path.join(path, folder, file_name)

    # 运行t-SNE降维
    for layer, data in data_epoch.items():
        layer_plt = kn_map_layer(data, label, layer, group_size=group_size)
        layer_plt.savefig(full_file_path + '_' + layer + '.png')
        print('Saved plot {}'.format(layer + '.png'))

    # 跑一张整体图，带不同标签
    # data = np.vstack([data for data in data_epoch.values()])
    # layer_plt = kn_map_layer(data, label, 'all', group_size=group_size)

    return os.path.join(path, folder)




# 利用t-sne对各层输入数据降维可视化,使用seaborn绘图
def kn_map_layer(data, label, layer='', group_size=25):
    '''
    :param data: 二维数据, (batch_size, feature_dim)
    :param label: 标量
    :param layer: str, 层名称
    :return: plot
    '''

    # data = data.cpu().detach().numpy()
    # 检查输入数据的维度
    if len(data.shape) != 2:
        raise ValueError("Data should be 2D array.")

    data = data.cpu().detach().numpy()
    # if len(label.shape) != 1:
    #     raise ValueError("Label should be 1D array.")

    # if data.shape[0] != label.shape[0]:
    #     raise ValueError("Data and label must have the same number of samples.")

    # 运行t-SNE降维
    tsne = TSNE(n_components=2, perplexity=min(group_size * 2 / 4, data.shape[0]), n_iter=1000)
    data_tsne = tsne.fit_transform(data)

    # 将降维后的数据和标签转换为DataFrame
    df = pd.DataFrame(data_tsne, columns=['Dim1', 'Dim2'])
    df['label'] = label

    # 使用Seaborn设置风格
    sns.set_theme(style="whitegrid")

    # 使用Seaborn设置调色板
    sns.set_palette("pastel")

    # 创建Seaborn散点图
    plt.figure()

    scatter = sns.scatterplot(x='Dim1', y='Dim2', hue='label', data=df, palette="deep")

    # 设置图表的标题

    plt.title(f'{layer} t-SNE')



    # 显示图表
    # plt.show()

    return plt





# 可视化错误标签
def plot_wrong_label(data, label, pred, epoch, folder='', pre='', path=None, max_samples=10):
    '''
    :param data: 二维数据, (batch_size, feature_dim)
    :param label: 一维数据, (batch_size, )
    :param pred: 一维数据, (batch_size, )
    :param epoch: int, epoch
    :param folder: str, 文件夹名称
    :param path: str, 路径
    :return: plot
    '''
    folder = folder + '/wrong_label{:03d}'.format(epoch)
    file_name = pre + '_' + 'epoch_{:03d}'.format(epoch)
    # 如果path不为None，则在path中创建文件夹

    if not os.path.exists(path + folder):
        os.makedirs(path + folder)
    full_file_path = os.path.join(path, folder, file_name)

    # 如果label是one-hot编码，则转换为标量
    if len(label.shape) > 1:
        label = np.argmax(label, axis=1)

    # 获取错误标签的索引
    wrong_idx = np.where(label != pred)[0]

    # 如果错误标签的数量超过最大样本数，则随机选择最大样本数个错误标签
    if len(wrong_idx) > max_samples:
        wrong_idx = np.random.choice(wrong_idx, max_samples, replace=False)

    # 选取错误标签的数据和标签
    data = data[wrong_idx]
    label = label[wrong_idx]
    pred = pred[wrong_idx]

    # 创建Seaborn图
    plt.figure()
    for i in range(len(data)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(data[i], cmap='gray', interpolation='none')
        plt.title(f'True: {label[i]}\nPred: {pred[i]}')
        plt.axis('off')

    # 保存图表
    plt.savefig(full_file_path + '.png')

    return os.path.join(path, folder)