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


class BaseEpoch:
    max_epoch = 100

    def __init__(self, name, model, loader, device, interval=1, bar=True):
        self.name = name
        self.model = model
        self.loader = tqdm(enumerate(loader), total=len(loader)) if bar else loader
        self.device = device
        self.interval = interval

    def run(self, epoch, *args, **kwargs):
        if epoch % self.interval == 0 or epoch == self.max_epoch:
            print('Running {} Epoch: {}\n'.format(self.name, epoch))
            self._run_epoch(epoch, *args, **kwargs)

    def _run_epoch(self, epoch, *args, **kwargs):
        raise NotImplementedError

    # 设置所有epoch的最大迭代次数
    @classmethod
    def set_max_epoch(cls, max_epoch):
        cls.max_epoch = max_epoch


class TrainEpoch(BaseEpoch):
    def __init__(self, model, loader, optimizer, criterion, device, scaler=None, interval=1, bar=True):
        super(TrainEpoch, self).__init__('Train', model, loader, device, interval, bar)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler

    def _run_epoch(self, epoch, *args, **kwargs):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in self.loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                with autocast():
                    outputs, _ = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            self.loader.set_description(f"Loss: {loss.item():.4f}")

        train_loss = running_loss / len(self.loader)
        train_accuracy = correct / total

        print('train_loss: %.3f, train_accuracy: %.3f' % (train_loss, train_accuracy))


class ValEpoch(BaseEpoch):
    def __init__(self, model, loader, criterion, device, plot_wrong, epoch=0, replace_label=True, interval=1, bar=True):
        super(ValEpoch, self).__init__('Val', model, loader, device, interval, bar)
        self.criterion = criterion
        self.plot_wrong = plot_wrong
        self.epoch = epoch
        self.replace_label = replace_label

    def _run_epoch(self, epoch, *args, **kwargs):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in self.loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs, _ = self.model(inputs)

                loss = self.criterion(outputs, targets)

                if batch_idx == 0 and self.plot_wrong > 0:
                    plot_wrong_label(inputs, targets, outputs, self.epoch, folder='wrong_output',
                                     max_samples=self.plot_wrong,
                                     replace_label=self.replace_label)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

        val_loss = running_loss / len(self.loader)
        val_accuracy = correct / total

        print('val_loss: %.3f, val_accuracy: %.3f' % (val_loss, val_accuracy))


class LIDComputeEpoch(BaseEpoch):
    def __init__(self, model, loader, device, num_class=10, group_size=15, interval=1, bar=True):
        super(LIDComputeEpoch, self).__init__('LID', model, loader, device, interval, bar)
        self.num_class = num_class
        self.group_size = group_size

    def _run_epoch(self, epoch, *args, **kwargs):
        if self.group_size < 2:
            return {'null': 0}, {'null': 0}
        self.model.eval()
        logits_list = defaultdict(dict)
        class_counts = [0] * self.num_class  # 记录每个类别收集的样本数
        with autocast():
            with torch.no_grad():
                for inputs, targets in self.loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    if len(targets.size()) > 1:
                        targets = torch.argmax(targets, dim=1)

                    outputs, logits = self.model(inputs)

                    for idx, target in enumerate(targets):
                        label = target.item()
                        if class_counts[label] < self.group_size:
                            for key, value in logits.items():
                                if key in logits_list[label]:
                                    logits_list[label][key] = torch.cat(
                                        (logits_list[label][key], value[idx].unsqueeze(0)),
                                        dim=0)
                                else:
                                    logits_list[label][key] = value[idx].unsqueeze(0)
                            class_counts[label] += 1

                        if all(count >= self.group_size for count in class_counts):
                            break

                    if all(count >= self.group_size for count in class_counts):
                        break

            # 计算每个类别的LID
            class_lidses = []
            for label, logits_per_class in logits_list.items():
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

        return lidses


class ExpressionSaveEpoch(BaseEpoch):
    def __init__(self, model, loader, device, path, folder, times, epoch='', num_class=10, group_size=15, interval=1,
                 bar=True):
        super(ExpressionSaveEpoch, self).__init__('Expression', model, loader, device, interval, bar)
        self.path = path
        self.folder = folder
        self.times = times
        self.epoch = epoch
        self.num_class = num_class
        self.group_size = group_size

    def _run_epoch(self, epoch, *args, **kwargs):
        self.folder = self.folder + '_{}'.format(kwargs['val_accuracy'])
        for self.time in range(self.times):
            current_path = os.path.join(self.path, self.folder, f'time_{self.times + 1}')
            print(f'Starting saving for time {self.times + 1} at {current_path}')
            # 如果current_path不存在，则创建一个文件夹，并保存一个所有类数量和名称信息的json文件
            if not os.path.exists(current_path):
                print('Create folder:', current_path)
                os.makedirs(current_path)
                # 保存类别信息
                class_info = {'num_class': self.num_class, 'class_name': self.loader.dataset.dataset.classes,
                              'epoch': self.epoch}
                json_file = os.path.join(current_path, 'class_info.json')
                with open(json_file, 'w') as f:
                    json.dump(class_info, f)

            self.model.eval()
            logits_list = defaultdict(dict)
            class_counts = [0] * self.num_class

            with autocast():
                with torch.no_grad():
                    for inputs, targets in self.loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        if len(targets.size()) > 1:
                            targets = torch.argmax(targets, dim=1)

                        outputs, logits = self.model(inputs)

                        for idx, target in enumerate(targets):
                            label = target.item()
                            if class_counts[label] < self.group_size:
                                for key, value in logits.items():
                                    if key in logits_list[label]:
                                        logits_list[label][key] = torch.cat(
                                            (logits_list[label][key], value[idx].unsqueeze(0)), dim=0)
                                    else:
                                        logits_list[label][key] = value[idx].unsqueeze(0)
                                class_counts[label] += 1

                            if all(count >= self.group_size for count in class_counts):
                                break

                        if all(count >= self.group_size for count in class_counts):
                            break

                # 保存每个类别的特征图到文件
                for label, logits_per_label in logits_list.items():
                    label_path = os.path.join(current_path, f'class_{label}')
                    os.makedirs(label_path, exist_ok=True)
                    for layer, feature_map in logits_per_label.items():
                        torch.save(feature_map.cpu(), os.path.join(label_path, f'{layer}.pt'))

            print('Save expression to:', current_path)


class NEComputeEpoch(BaseEpoch):
    def __init__(self, model, loader, device, num_class=10, group_size=15, interval=1, bar=True):
        super(NEComputeEpoch, self).__init__('NE', model, loader, device, interval, bar)
        self.num_class = num_class
        self.group_size = group_size

    def _run_epoch(self, epoch, *args, **kwargs):
        if self.group_size < 2:
            return {'null': 0}
        self.model.eval()
        logits_list = defaultdict(dict)
        class_counts = [0] * self.num_class  # 记录每个类别收集的样本数

        with torch.no_grad():
            for inputs, targets in self.loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if len(targets.size()) > 1:
                    targets = torch.argmax(targets, dim=1)

                outputs, logits = self.model(inputs)

                for idx, target in enumerate(targets):
                    label = target.item()
                    if class_counts[label] < self.group_size:
                        for key, value in logits.items():
                            if key in logits_list[label]:
                                logits_list[label][key] = torch.cat((logits_list[label][key], value[idx].unsqueeze(0)),
                                                                    dim=0)
                            else:
                                logits_list[label][key] = value[idx].unsqueeze(0)
                        class_counts[label] += 1

                    if all(count >= self.group_size for count in class_counts):
                        break

                if all(count >= self.group_size for count in class_counts):
                    break

        # 计算所有层的知识熵
        ne_dict = defaultdict(list)
        for label, logits_per_class in logits_list.items():
            for key, value in logits_per_class.items():
                ne_dict[key].append(knowledge_entropy(value))
            # ne_dict[key] = np.mean(ne_dict[key])

        print('ne_compute complete')

        return {key: np.mean(values) for key, values in ne_dict.items()}  # ne_dict


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
