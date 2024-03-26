# -*- CODING: UTF-8 -*-
# @time 2024/3/25 19:26
# @Author tyqqj
# @File epochs.py
# @
# @Aim
import json
import os
from collections import defaultdict

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

import numpy as np

from LID import get_lids_batches
from know_entropy import knowledge_entropy
from utils.BOX import logbox
from utils.plotfn import kn_map, plot_wrong_label

plot_kn_map = logbox.log_artifact_autott(kn_map)
plot_wrong_label = logbox.log_artifact_autott(plot_wrong_label)


def train_epoch(model, data_loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # print('\n')
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, (inputs, targets) in progress_bar:
        # print('start')
        # print(inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
                outputs, _ = model(inputs)
                # print('outputs:', outputs.shape, 'targets:', targets.shape)
                loss = criterion(outputs, targets)
            # print(f"{batch_idx}: loss:", loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # .step()
            scaler.update()
        else:
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # 更新进度条显示的信息
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
        # print('end')
    # print('\n')
    train_loss = running_loss / len(data_loader)
    train_accuracy = correct / total

    return train_loss, train_accuracy


def val_epoch(model, data_loader, criterion, device, plot_wrong, epoch=0, replace_label=True):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, _ = model(inputs)

            loss = criterion(outputs, targets)

            if batch_idx == 0 and plot_wrong > 0:
                # _, predicted = torch.max(outputs.data, 1).cpu().detach()
                plot_wrong_label(inputs, targets, outputs, epoch, folder='wrong_output', max_samples=plot_wrong,
                                 replace_label=replace_label)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    val_loss = running_loss / len(data_loader)
    val_accuracy = correct / total
    return val_loss, val_accuracy


def lid_compute_epoch(model, data_loader, device, num_class=10, group_size=15):
    '''
    计算LID
    :param model:
    :param data_loader: 使用训练集
    :param device:
    :param group_size: 计算LID时的每类取数据量, 在分类任务时现阶段使用，后序将计算Y密度/X密度代替
    '''
    if group_size < 2:
        return {'null': 0}, {'null': 0}
    model.eval()
    logits_list = defaultdict(dict)
    # 存储每个类别的logits,defaultdict是一个字典，当字典里的key不存在但被查找时，返回的不是keyEror而是一个默认值
    class_counts = [0] * num_class  # 记录每个类别收集的样本数
    with autocast():
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # 如果targets不是一维的，就转换成一维的softmax
                if len(targets.size()) > 1:
                    targets = torch.argmax(targets, dim=1)

                # optimizer.zero_grad()
                outputs, logits = model(inputs)  # 假设模型返回的最后一个元素是logits

                # 遍历batch中的每个样本
                for idx, target in enumerate(targets):
                    label = target.item()
                    # 检查是否已经有足够的样本
                    if class_counts[label] < group_size:
                        # 此处logits_list[label]应该是一个字典,结构为{'layer_name', data_tensor}应在data_tensor处拼接新的logits
                        for key, value in logits.items():
                            # logits_list
                            if key in logits_list[label]:
                                logits_list[label][key] = torch.cat((logits_list[label][key], value[idx].unsqueeze(0)),
                                                                    dim=0)
                            else:
                                logits_list[label][key] = value[idx].unsqueeze(0)
                        class_counts[label] += 1
                    # 如果每个类别都收集到了足够的样本，就退出
                    if all(count >= group_size for count in class_counts):
                        break
                # 如果每个类别都收集到了足够的样本，就退出
                if all(count >= group_size for count in class_counts):
                    break

        # 计算每个类别的LID
        class_lidses = []
        for label, logits_per_class in logits_list.items():
            # 假设get_lids_batches是计算LID的函数
            # 这里需要将logits转换为tensor，因为它目前是一个列表
            # tensor_logits = torch.stack(logits_per_class)
            # print(logits_per_class.shape)
            class_lidses.append(get_lids_batches(logits_per_class))

        # 求class_lidses的平均值
        lidses = {key: 0 for key in class_lidses[0].keys()}
        for a_lids in class_lidses:
            for key, value in a_lids.items():
                lidses[key] += value
        for key in lidses.keys():
            lidses[key] = lidses[key] / len(class_lidses)
            if np.isnan(lidses[key]):
                lidses[key] = 0

    #
    return lidses, logits_list


def expression_save_epoch(model, data_loader, device, path, folder, times, epoch='', num_class=10, group_size=15,
                          shuffle=False):
    '''
    保存定量所有数据的各层特征图到文件夹
    :param model: 模型
    :param data_loader: 使用训练集
    :param device: 设备
    :param path: 保存路径，格式为model_valacc_noise
    :param folder: 保存的子文件夹名
    :param times: 重复采样的次数
    :param epoch: 当前的epoch
    :param num_class: 类别数
    :param group_size: 每类取数据量
    :param shuffle: 是否打乱数据顺序
    '''
    for time in range(times):
        current_path = os.path.join(path, folder, f'time_{time + 1}')
        print(f'Starting saving for time {time + 1} at {current_path}')
        # 如果current_path不存在，则创建一个文件夹，并保存一个所有类数量和名称信息的json文件
        if not os.path.exists(current_path):
            print('Create folder:', current_path)
            os.makedirs(current_path)
            # 保存类别信息
            class_info = {'num_class': num_class, 'class_name': data_loader.dataset.dataset.classes, 'epoch': epoch}
            json_file = os.path.join(current_path, 'class_info.json')
            with open(json_file, 'w') as f:
                json.dump(class_info, f)

        model.eval()
        logits_list = defaultdict(dict)
        class_counts = [0] * num_class  # 记录每个类别收集的样本数

        with autocast():
            with torch.no_grad():
                for inputs, targets in data_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    if len(targets.size()) > 1:
                        targets = torch.argmax(targets, dim=1)

                    outputs, logits = model(inputs)

                    for idx, target in enumerate(targets):
                        label = target.item()
                        if class_counts[label] < group_size:
                            for key, value in logits.items():
                                if key in logits_list[label]:
                                    logits_list[label][key] = torch.cat(
                                        (logits_list[label][key], value[idx].unsqueeze(0)), dim=0)
                                else:
                                    logits_list[label][key] = value[idx].unsqueeze(0)
                            class_counts[label] += 1

                        if all(count >= group_size for count in class_counts):
                            break

                    if all(count >= group_size for count in class_counts):
                        break

            # 保存每个类别的特征图到文件
            for label, logits_per_label in logits_list.items():
                label_path = os.path.join(current_path, f'class_{label}')
                os.makedirs(label_path, exist_ok=True)
                for layer, feature_map in logits_per_label.items():
                    torch.save(feature_map.cpu(), os.path.join(label_path, f'{layer}.pt'))

        print('Save expression to:', current_path)


def ne_compute_epoch(model, data_loader, device, num_class=10, group_size=15):
    '''
    计算知识熵并返回
    :param model: 模型
    :param data_loader: 使用训练集
    :param device: 设备
    :param num_class: 类别数
    :param group_size: 每类取数据量
    '''

    model.eval()
    logits_list = defaultdict(dict)
    class_counts = [0] * num_class  # 记录每个类别收集的样本数

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if len(targets.size()) > 1:
                targets = torch.argmax(targets, dim=1)

            outputs, logits = model(inputs)

            for idx, target in enumerate(targets):
                label = target.item()
                if class_counts[label] < group_size:
                    for key, value in logits.items():
                        if key in logits_list[label]:
                            logits_list[label][key] = torch.cat(
                                (logits_list[label][key], value[idx].unsqueeze(0)), dim=0)
                        else:
                            logits_list[label][key] = value[idx].unsqueeze(0)
                    class_counts[label] += 1

                if all(count >= group_size for count in class_counts):
                    break

            if all(count >= group_size for count in class_counts):
                break

    # 计算所有层的知识熵
    ne_dict = defaultdict(list)
    for label, logits_per_class in logits_list.items():
        for key, value in logits_per_class.items():
            ne_dict[key].append(knowledge_entropy(value))
        # ne_dict[key] = np.mean(ne_dict[key])

    print('ne_compute complete')


    return {key: np.mean(values) for key, values in ne_dict.items()} #ne_dict


def plot_kmp(epoch, logits_list, model_name='', noise_ratio=0.0, folder='kn_map'):
    logits = {}  # defaultdict(torch.Tensor)
    labels = []
    # logits_list:{label: {layer_name: data_tensor_of_group_size_logits}
    for label, logits_per_class in logits_list.items():
        # 拼接每个类别的logits
        for key, value in logits_per_class.items():
            if label not in labels:
                labels.extend([label] * value.shape[0])  # labels.extend([label] * value.shape[0])
            if key in logits:
                logits[key] = torch.cat((logits[key], value), dim=0)

            else:
                # print(key)
                logits[key] = value

                # print([label] * value.shape[0])
    # print(logits[key].shape, len(labels))
    with autocast():
        plot_kn_map(logits, labels, epoch=epoch, folder=folder,
                    pre=model_name + '_' + str(noise_ratio) + '_epoch_' + str(epoch + 1))


# 将knowledge的字典转换为json并保存提交
@logbox.log_artifact_autott
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
