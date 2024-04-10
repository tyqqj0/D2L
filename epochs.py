# -*- CODING: UTF-8 -*-
# @time 2024/3/25 19:26
# @Author tyqqj
# @File epochs.py
# @
# @Aim
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from LID import get_lids_batches
from MOF import compute_pca_correct, compute_vec_corr, mof
from MOF import bkc
from know_entropy import knowledge_entropy, compute_knowledge, FeatureMapSimilarity  # n, knowledge_entropy2
from utils.BOX import logbox
from utils.plotfn import kn_map, plot_wrong_label, plot_images

from model.clusters.BasicCluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch, GMM, \
    TsneGMM, TsneKMeans

plot_kn_map = logbox.log_artifact_autott(kn_map)
plot_wrong_label = logbox.log_artifact_autott(plot_wrong_label)
plot_images = logbox.log_artifact_autott(plot_images)


class BaseEpoch:
    max_epoch = 100

    def __init__(self, name, model, loader, device, interval=1, bar=True):
        self.name = name
        self.model = model
        self.loaders = loader
        self.loader = None
        self.device = device
        self.interval = interval
        self.bar = bar

    def run(self, epoch, *args, **kwargs):
        if epoch % self.interval == 0 or epoch == self.max_epoch:

            if self.bar:
                # 使用 tqdm.write 来打印信息
                # tqdm.write(f'\nRunning {self.name} Epoch: {epoch}\n')
                self.loader = tqdm(enumerate(self.loaders), total=len(self.loaders))

            else:
                print('Running {} Epoch: {}'.format(self.name, epoch))  # 打印信息移动到这里
                self.loader = enumerate(self.loaders)  #
                pass

            if len(self.loaders) == 0:  # 注意这里应该是 self.loaders 而不是 self.loader
                raise ValueError("Loader is empty")

            result = self._run_epoch(epoch, *args, **kwargs)

            if self.bar:
                self.loader.close()  # 确保关闭 tqdm 进度条
                self.loader.clear(nolock=False)
                self.loader.refresh()
            return result
        else:
            return self._default_return()

    def _run_epoch(self, epoch, *args, **kwargs):
        raise NotImplementedError

    def _default_return(self):
        return [None, None]

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
                    outputs, logits = self.model(inputs)
                    l = logits['layer4']
                    loss = self.criterion(outputs, targets, hi=l)
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

            self.loader.set_description(f"Train Loss: {loss.item():.4f}")

        train_loss = running_loss / len(self.loader)
        train_accuracy = correct / total

        # print('train_loss: %.3f, train_accuracy: %.3f' % (train_loss, train_accuracy))
        return train_loss, train_accuracy


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
                self.loader.set_description(f"Val Loss: {loss.item():.4f}")

        val_loss = running_loss / len(self.loader)
        val_accuracy = correct / total

        # print('val_loss: %.3f, val_accuracy: %.3f' % (val_loss, val_accuracy))
        return val_loss, val_accuracy


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
                for batch_idx, (inputs, targets) in self.loader:
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

        return lidses, logits_list


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
            # print(f'Starting saving for time {self.times + 1} at {current_path}')
            # 如果current_path不存在，则创建一个文件夹，并保存一个所有类数量和名称信息的json文件
            if not os.path.exists(current_path):
                # print('Create folder:', current_path)
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
                    for batch_idx, (inputs, targets) in self.loader:
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

            # print('Save expression to:', current_path)


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
            for batch_idx, (inputs, targets) in self.loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if len(targets.size()) > 1:
                    targets = torch.argmax(targets, dim=1)

                outputs, logits = self.model(inputs)

                for idx, target in enumerate(targets):
                    label = target.item()
                    if class_counts[label] < self.group_size:
                        for key, value in logits.items():
                            # print(value.shape)
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
        for label, logits_per_class in tqdm(logits_list.items(), desc='Computing knowledge entropy'):
            for key, value in logits_per_class.items():
                # print(value.shape)
                # 将(n, C, h, w)的数据转换为C, n, h, w的
                # value = value.permute(1, 0, 2, 3)
                ne_dict[key].append(knowledge_entropy(value))
            # ne_dict[key] = np.mean(ne_dict[key])

        # print('ne_compute complete')

        return {key: np.mean(values) for key, values in ne_dict.items()}  # ne_dict


# 主成分修正样本标签修正
class PCACorrectEpoch(BaseEpoch):
    def __init__(self, model, loader, device, num_class=10, group_size=15, interval=1, bar=True):
        super(PCACorrectEpoch, self).__init__('PCA', model, loader, device, interval, bar)
        self.num_class = num_class
        self.group_size = group_size

    def _run_epoch(self, epoch, *args, **kwargs):
        if self.group_size < 2:
            return {'null': 0}
        self.model.eval()
        # 获取各层的logits
        logits_list = defaultdict(dict)
        class_counts = [0] * self.num_class
        with torch.no_grad():
            for batch_idx, (inputs, targets) in self.loader:
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

            # 计算所有层特征向量图的主成分({label, {layer, (C, C, H, W)}})
            pca_dict = defaultdict(dict)
            for label, logits_per_class in tqdm(logits_list.items(), desc='Computing PCA'):
                for key, value in logits_per_class.items():
                    # 暂时只取最后一层
                    if key == 'layer4':
                        # print(label, key)
                        # pca_dict[label][key] = compute_knowledge(value)
                        _, pca_dict[label][key] = compute_knowledge(value)

            fimttt = FeatureMapSimilarity(method='cosine')

            # 计算要修正的主成分
            layert = 'layer4'
            pca_corrects = {}
            max_cos = np.zeros((2, self.num_class, self.num_class))
            for cl1 in range(0, self.num_class):
                pca_corrects[cl1] = {}
                for cl2 in range(0, self.num_class):
                    if cl1 == cl2:
                        continue
                    pca_corrects[cl1][cl2] = []
                    # 先用前两个类为例
                    # print('cl1, cl2', cl1, cl2)
                    class2to1, confdt, max_cor = compute_pca_correct(pca_dict[cl1][layert], pca_dict[cl2][layert],
                                                                     fimttt)
                    # print('confdt', confdt)
                    max_cos[0, cl1, cl2] = max_cor
                    max_cos[1, cl1, cl2] = confdt

                    if confdt <= 0.5:
                        continue
                    # 获取2中与该成分对齐的样本
                    logits2 = [logits_list[cl2][layert], logits_list[cl2]['image']]
                    # logits2 = logits2
                    # print(logits2.shape)
                    for i in range(logits2[0].shape[0]):
                        # print(torch.max(logits2[i]))
                        #     # print(class2to1.shape, logits2[i].shape)
                        simttt = compute_vec_corr(class2to1, logits2[0][i].view(logits2[0][i].shape[0]))
                        print(simttt)
                        if abs(simttt * confdt * 100) > 0.65:
                            pca_corrects[cl1][cl2].append(
                                logits2[1][i].cpu().view(logits2[1][i].shape[0], -1))  # 应归为cl1的样本
            # 打印 保留两位小数
            # np.set_printoptions(precision=2)

            np.set_printoptions(precision=2)
            print(np.array2string(max_cos[0], formatter={'float_kind': lambda x: "%.2f" % x}))
            print(np.array2string(max_cos[1], formatter={'float_kind': lambda x: "%.2f" % x}))

            # 保存类2中实际标签为1的样本图片
            # 找到一个置信度最高的
            for cl1 in range(0, self.num_class):
                for cl2 in range(0, self.num_class):
                    if cl1 == cl2:
                        continue
                    if len(pca_corrects[cl1][cl2]) > 0:
                        plot_images(pca_corrects[cl1][cl2], epoch, folder='pca_correct_{}'.format(epoch),
                                    pre=f'pca_correct_{cl2}to{cl1}')
                        # return
            # plot_images(pca_corrects[1][2], epoch, folder='pca_correct', pre='pca_correct2to1')


class PCAFindEpoch(BaseEpoch):
    def __init__(self, model, loader, device, num_class=10, group_size=15, interval=1, bar=True):
        super(PCACorrectEpoch, self).__init__('PCA', model, loader, device, interval, bar)
        self.num_class = num_class
        self.group_size = group_size

    def _run_epoch(self, epoch, *args, **kwargs):
        if self.group_size < 2:
            return {'null': 0}
        self.model.eval()
        # 获取各层的logits
        logits_list = defaultdict(dict)
        class_counts = [0] * self.num_class
        with torch.no_grad():
            for batch_idx, (inputs, targets) in self.loader:
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

            # 计算所有层特征向量图的主成分({label, {layer, (C, C, H, W)}})
            pca_dict = defaultdict(dict)
            for label, logits_per_class in tqdm(logits_list.items(), desc='Computing PCA'):
                for key, value in logits_per_class.items():
                    # 暂时只取最后一层
                    if key == 'layer4':
                        # print(label, key)
                        # pca_dict[label][key] = compute_knowledge(value)
                        _, pca_dict[label][key] = compute_knowledge(value)

            fimttt = FeatureMapSimilarity(method='cosine')

            # 计算要修正的主成分
            layert = 'layer4'
            pca_corrects = {}
            for cl1 in range(0, self.num_class):
                pca_corrects[cl1] = {}
                for cl2 in range(0, self.num_class):
                    if cl1 == cl2:
                        continue
                    pca_corrects[cl1][cl2] = []

                    # 获取2中与该成分对齐的样本
                    logits2 = [logits_list[cl2][layert], logits_list[cl2]['image']]
                    # logits2 = logits2
                    # print(logits2.shape)
                    for i in range(logits2[0].shape[0]):
                        # print(torch.max(logits2[i]))
                        #     # print(class2to1.shape, logits2[i].shape)
                        simttt = compute_vec_corr(pca_dict, logits2[0][i].view(logits2[0][i].shape[0]))
                        print(simttt)
                        if abs(simttt) > 0.65:
                            pca_corrects[cl1][cl2].append(
                                logits2[1][i].cpu().view(logits2[1][i].shape[0], -1))  # 应归为cl1的样本
            # 打印 保留两位小数
            # np.set_printoptions(precision=2)

            np.set_printoptions(precision=2)
            # print(np.array2string(max_cos[0], formatter={'float_kind': lambda x: "%.2f" % x}))
            # print(np.array2string(max_cos[1], formatter={'float_kind': lambda x: "%.2f" % x}))

            # 保存类2中实际标签为1的样本图片
            # 找到一个置信度最高的
            for cl1 in range(0, self.num_class):
                for cl2 in range(0, self.num_class):
                    if cl1 == cl2:
                        continue
                    if len(pca_corrects[cl1][cl2]) > 0:
                        plot_images(pca_corrects[cl1][cl2], epoch, folder='pca_correct_{}'.format(epoch),
                                    pre=f'pca_correct_{cl2}to{cl1}')
                        # return
            # plot_images(pca_corrects[1][2], epoch, folder='pca_correct', pre='pca_correct2to1')


class ClusterBackwardEpoch(BaseEpoch):
    def __init__(self, model, loader, criterion, cluster_model, device, num_class=10, group_size=15, interval=1,
                 bar=True):
        super(ClusterBackwardEpoch, self).__init__('Cluster', model, loader, device, interval, bar)
        self.num_class = num_class
        self.group_size = group_size
        self.layers = ['layer4', 'label']
        if cluster_model == 'kmeans' or cluster_model == 'KMeans':
            self.cluster_model = KMeans()
        elif cluster_model == 'dbscan' or cluster_model == 'DBSCAN':
            self.cluster_model = DBSCAN()
        elif cluster_model == 'agglomerative' or cluster_model == 'Agglomerative':
            self.cluster_model = AgglomerativeClustering()
        elif cluster_model == 'spectral' or cluster_model == 'Spectral':
            self.cluster_model = SpectralClustering()
        elif cluster_model == 'birch' or cluster_model == 'Birch':
            self.cluster_model = Birch()
        elif cluster_model == 'gmm' or cluster_model == 'GMM':
            self.cluster_model = GMM()
        elif cluster_model == 'tsne_gmm' or cluster_model == 'TsneGMM':
            self.cluster_model = TsneGMM()
        elif cluster_model == 'tsne_kmeans' or cluster_model == 'TsneKMeans':
            self.cluster_model = TsneKMeans()
        else:
            raise ValueError('Unknown cluster model')
        print(self.cluster_model)
        self.criterion = criterion

    def _get_logits(self, epoch):
        if self.group_size < 2:
            return {'null': 0}
        self.model.eval()
        # 获取各层的logits
        # logits_list = defaultdict(lambda: defaultdict(torch.tensor))
        # class_counts = [0] * self.num_class
        logits_list = defaultdict(torch.tensor)
        class_counts = [0] * self.num_class
        with torch.no_grad():
            for batch_idx, (inputs, targets) in self.loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if len(targets.size()) > 1:
                    targets = torch.argmax(targets, dim=1)

                outputs, logits = self.model(inputs)
                # logits['label'] = targets  # TODO
                logits['label'] = targets.unsqueeze(1)

                for idx, target in enumerate(targets):
                    label = target.item()
                    if class_counts[label] < self.group_size:
                        for layer, value in logits.items():
                            if layer in logits_list:
                                logits_list[layer] = torch.cat((logits_list[layer], value[idx].unsqueeze(0)), dim=0)
                            else:
                                logits_list[layer] = value[idx].unsqueeze(0)
                        class_counts[label] += 1

                    if all(count >= self.group_size for count in class_counts):
                        return dict(logits_list)

    def _run_epoch(self, epoch, *args, **kwargs):
        logits_list = self._get_logits(epoch)
        # 从Hi和Hi-1中循环算更正方向向量, 从倒数第二层与倒数第一层开始往前
        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i - 1]
            next_layer = self.layers[i]
            print(layer, next_layer)
            # print(logits_list[next_layer].shape)
            # 对最后一层聚类
            cluster_labels_next = self.cluster_model.fit(logits_list[next_layer], logits_list['label'])

            # 计算获取了几个类, 用unique去重
            next_layer_classes = torch.unique(cluster_labels_next).int().tolist()

            # 显示聚类结果标签与原标签有多少一样
            # if cluster_labels_next.shape != logits_list['label'].shape:
            #     # 如果形状不同,可以考虑对其中一个进行调整
            #     # 例如,如果cluster_labels_next是一维张量,而logits_list['label']是二维张量,可以将cluster_labels_next扩展为二维张量
            #     cluster_labels_next1 = cluster_labels_next.view(-1, 1)
            # numttt = torch.sum(cluster_labels_next1 == logits_list['label'])
            # print(numttt, cluster_labels_next1.shape, logits_list['label'].shape)

            # 打印各获取了几个类
            # print('cluster_labels_next:', len(next_layer_classes))

            cluster_per_label = {}
            # 按照最后一层分出来的类遍历类别分别对上一层聚类
            pbar = tqdm(next_layer_classes, desc=f'Clustering')
            for next_label in pbar:
                # 获取上一层的特征向量
                logits_list_this = logits_list[layer][
                    cluster_labels_next == next_label]  # 如果不行就写成torch的形式找label对应的数据的layer层数据

                # print(logits_list_this.shape)
                # 对上一层聚类
                cluster_labels_this = self.cluster_model.fit(logits_list_this)
                # 计算获取了几个类, 用unique去重
                classes = torch.unique(cluster_labels_this).int().tolist()
                # 打印各获取了几个类
                # print('cluster_labels_this:', len(classes))
                # tqdm.write(f"Cluster {next_label}: {len(classes)} classes")
                # pbar.set_description(f'Clustering (Cluster {next_label}: {len(classes)} classes)')
                pbar.set_postfix(cluster=next_label, classes=len(classes))

                # 将所有数据按照类别重排(h1, h2, h3) (1, 1, 2)
                # idx = np.argsort(cluster_labels_this)
                # logits_list_this = logits_list_this[idx]
                # cluster_labels_this = cluster_labels_this[idx]

                # 记录改类别下所有layer层数据(n,M)以及其对应layer层聚类标签(n)
                cluster_per_label[next_label] = [logits_list_this, cluster_labels_this]
                # print(logits_list[layer].shape)

            vec_allt = {}
            val_allt = {}
            # 对每个标签的数据做特征值分解
            for next_label in tqdm(next_layer_classes, desc='MOF'):
                # next_label = next_label.int()
                vecs_this, val_this = mof(cluster_per_label[next_label][0], cluster_per_label[next_label][1],
                                          num=len(next_layer_classes) + 5)
                # print(vecs_this.shape)
                # 此时每个vec_this是一个特征向量
                vec_allt[next_label] = vecs_this
                val_allt[next_label] = val_this
            # 破碎知识筛选
            corrcetv = bkc(vec_allt, val_allt, next_layer_classes)

            # 计算各类修正后成分与数据的相似度
            if kwargs['check']:
                for next_label in next_layer_classes:
                    vec = corrcetv[next_label]  # (num, M)
                    # print(logits_list[layer].shape)

                    datatt = logits_list[layer][cluster_labels_next == next_label, :]
                    datatt = datatt.view(datatt.shape[0], -1)
                    # vec = vec / torch.norm(vec, dim=1, keepdim=True)
                    datatt = datatt / torch.norm(datatt, dim=1, keepdim=True)
                    # print(torch.norm(datatt, dim=1).shape)
                    # datatt = (datatt - datatt.mean(dim=1, keepdim=True)) / datatt.std(dim=1, keepdim=True)
                    # vec = (vec - vec.mean(dim=1, keepdim=True)) / vec.std(dim=1, keepdim=True)
                    # print("logits_list[layer].shape:", logits_list[layer].shape)
                    # print("cluster_labels_next.shape:", cluster_labels_next.shape)
                    # print("datatt.shape:", datatt.shape)
                    # print("vec.shape:", vec.shape)

                    # 计算相似程度
                    corrst = torch.matmul(datatt, vec.T)  # (n, num)
                    corrst = torch.max(abs(corrst), dim=1)[0]  # (n)

                    # 计算 datatt 和 vec 的模
                    # datatt_norm = torch.norm(datatt, dim=1, keepdim=True)
                    # vec_norm = torch.norm(vec, dim=1, keepdim=True)
                    #
                    # # 除以模进行归一化
                    # corrst = corrst / (datatt_norm * vec_norm.t())

                    print(next_label, corrst)

            # 将corrcetv转换为tensor
            # cn = torch.tensor([corrcetv[key] for key in corrcetv.keys()])
            # 获取字典的键列表
            keys = list(corrcetv.keys())

            # 使用列表推导式获取字典的值列表
            cn_list = [corrcetv[key] for key in keys]

            # 使用torch.stack将值列表拼接成一个张量
            cn = torch.stack(cn_list)

        self.criterion.set_cn(cn)

        return cn


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
