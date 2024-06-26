# -*- CODING: UTF-8 -*-
# @time 2024/4/8 13:02
# @Author tyqqj
# @File BasicCluster.py
# @
# @Aim
import os
from collections import defaultdict

# import os
import matplotlib.pyplot as plt
# os.environ['OMP_NUM_THREADS'] = '1'
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster')
import sklearn.manifold
import sklearn.cluster
from sklearn.mixture import GaussianMixture
import torch

# import pandas as pd
# import torchvision


__all__ = ['KMeans', 'DBSCAN', 'AgglomerativeClustering', 'SpectralClustering', 'Birch', 'GMM', 'TsneGMM', 'TsneKMeans']


def _tsne(x):
    # 使用t-SNE降维
    tsne = sklearn.manifold.TSNE(n_components=2)
    x = tsne.fit_transform(x)
    return x


def _pca(x):
    # 使用PCA降维
    pca = sklearn.decomposition.PCA(n_components=2)
    x = pca.fit_transform(x)
    return x


class BasicCluster:
    def __init__(self, model, num_features=None, plot_method='tsne'):
        self.num_features = num_features
        self.device = 0
        self.model = model
        self.fitted = False
        self.cluster_result = None

    def __str__(self):
        # 返回模型名称
        return self.model.__class__.__name__

    def predict(self, x):
        if not self.fitted:
            raise ValueError('Model not fitted yet')
        if isinstance(x, torch.Tensor):
            self.device = x.device
            x = x.cpu().detach().numpy()
        # 返回单个数据的聚类结果
        result = self.model.predict(x)
        result = torch.tensor(result, device=self.device)
        return result

    def _fit(self, x):
        self.model.fit(x)
        # 拟合并返回所有数据的聚类结果
        # 获得原数据的聚类结果(M)
        result = self.model.predict(x)
        # print('model {} fitted'.format(self.model.__class__.__name__))
        return result

    def fit(self, x1, labels=None):
        # x(n, M)
        # print(x.shape)
        x = x1
        self.num_features = x.shape[1]
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 4:
                x = x.view(x.shape[0], -1)
            self.device = x.device
            x = x.cpu().detach().numpy()

        cluster_labels = self._fit(x)

        if labels is not None:
            # 创建一个字典,用于存储每个聚类中出现次数最多的原始标签
            label_count = defaultdict(lambda: defaultdict(int))
            for cluster_label, original_label in zip(cluster_labels, labels):
                label_count[cluster_label][original_label] += 1

            # 将每个聚类映射到出现次数最多的原始标签
            label_map = {}
            for cluster_label, count_dict in label_count.items():
                label_map[cluster_label] = max(count_dict, key=count_dict.get)

            # 将聚类结果映射到出现次数最多的原始标签
            mapped_labels = [label_map[cluster_label] for cluster_label in cluster_labels]
            mapped_labels = torch.tensor(mapped_labels, device=self.device)
            self.cluster_result = mapped_labels
        else:
            cluster_labels = torch.tensor(cluster_labels, device=self.device)
            self.cluster_result = cluster_labels

            # print('model {} fitted'.format(self.model.__class__.__name__))
        return self.cluster_result

    # 设置全局的保存装饰器
    @staticmethod
    def set_save_wrapper(save_wrapper):
        BasicCluster.save_wrapper = save_wrapper

    @property
    def plot(self, x, folder, pre, epoch, path=None):
        # 应用保存装饰器直
        return self.save_wrapper(self._plot)(x, folder, pre, epoch, path)

    def _plot(self, x, folder, pre, epoch, path=None):
        # 绘制聚类结果
        if not self.fitted:
            raise ValueError('Model not fitted yet')
        result = self.predict(x)
        # 降维
        if self.num_features > 2:
            x = self._reduce_dim(x)

        # 绘制
        plt.scatter(x[:, 0], x[:, 1], c=result)

        file_name = pre + '_' + 'epoch_{:03d}'.format(epoch)
        # 如果path不为None，则在path中创建文件夹

        if not os.path.exists(path + folder):
            os.makedirs(path + folder)
        full_file_path = os.path.join(path, folder, file_name)

        plt.savefig(full_file_path + '.png')
        plt.close()

        return os.path.join(path, folder)

    def _reduce_dim(self, x):
        # 降维
        if self.num_features > 2:
            if self.num_features > 50:
                x = _tsne(x)
            else:
                x = _pca(x)
        return x


class KMeans(BasicCluster):
    def __init__(self, num_features=None, device=0, num_clusters=10):
        model = sklearn.cluster.KMeans(n_clusters=num_clusters)
        super(KMeans, self).__init__(model, num_features)
        self.num_clusters = num_clusters


class DBSCAN(BasicCluster):
    def __init__(self, num_features=None, device=0, eps=0.5, min_samples=5):
        model = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)
        super(DBSCAN, self).__init__(model, num_features)

    # dbscan没有predict方法，所以重写fit方法
    def _fit(self, x):
        # x(n, M)
        # print(x.shape)

        result = self.model.fit_predict(x)

        # print('model {} fitted'.format(self.model.__class__.__name__))
        return result


# class HDBSCAN(BasicCluster):
#     def __init__(self, num_features=None, device=0, min_cluster_size=5):
#         model = sklearn.cluster.HDBSCAN(min_cluster_size=min_cluster_size)
#         super(HDBSCAN, self).__init__(model, num_features)


class AgglomerativeClustering(BasicCluster):
    def __init__(self, num_features=None, device=0, n_clusters=10):
        model = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters)
        super(AgglomerativeClustering, self).__init__(model, num_features)


class SpectralClustering(BasicCluster):
    def __init__(self, num_features=None, device=0, n_clusters=10):
        model = sklearn.cluster.SpectralClustering(n_clusters=n_clusters)
        super(SpectralClustering, self).__init__(model, num_features)


class Birch(BasicCluster):
    def __init__(self, num_features=None, device=0, n_clusters=10):
        model = sklearn.cluster.Birch(n_clusters=n_clusters)
        super(Birch, self).__init__(model, num_features)


class GMM(BasicCluster):
    def __init__(self, num_features=None, device=0, n_clusters=10):
        model = GaussianMixture(n_components=n_clusters)
        super(GMM, self).__init__(model, num_features)


class TsneGMM(BasicCluster):
    def __init__(self, num_features=None, device=0, n_clusters=10):
        model = GaussianMixture(n_components=n_clusters)
        super(TsneGMM, self).__init__(model, num_features)
        self.tsne = sklearn.manifold.TSNE(n_components=2)

    def _fit(self, x):
        # x(n, M)
        # print(x.shape)
        # print(x.shape)
        if x.shape[1] != 1:
            x = self.tsne.fit_transform(x)
        self.model.fit(x)
        # 拟合并返回所有数据的聚类结果

        result = self.model.predict(x)

        # print('model {} fitted'.format(self.model.__class__.__name__))
        return result


class TsneKMeans(BasicCluster):
    def __init__(self, num_features=None, device=0, num_clusters=10):
        model = sklearn.cluster.KMeans(n_clusters=num_clusters)
        super(TsneKMeans, self).__init__(model, num_features)
        self.tsne = sklearn.manifold.TSNE(n_components=2)

    def _fit(self, x):
        # x(n, M)
        # print(x.shape)

        x = self.tsne.fit_transform(x)
        result = self.model.fit(x)

        # 拟合并返回所有数据的聚类结果

        result = self.model.predict(x)

        return result
