# -*- CODING: UTF-8 -*-
# @time 2024/3/20 18:08
# @Author tyqqj
# @File plotfn.py
# @
# @Aim


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

    # 创建Seaborn条形图
    plt.figure()
    barplot = sns.barplot(x='Layers', y='Values', data=data, ci=None)

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
