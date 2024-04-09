# -*- CODING: UTF-8 -*-
# @time 2024/4/8 13:02
# @Author tyqqj
# @File BasicCluster.py
# @
# @Aim
import os

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


__all__ = ['KMeans', 'DBSCAN', 'AgglomerativeClustering', 'SpectralClustering', 'Birch', 'GMM']


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

    def fit(self, x):
        # x(n, M)
        # print(x.shape)
        self.num_features = x.shape[1]
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 4:
                x = x.view(x.shape[0], -1)
            self.device = x.device
            x = x.cpu().detach().numpy()
        self.model.fit(x)
        self.fitted = True
        # 拟合并返回所有数据的聚类结果
        # 获得原数据的聚类结果(M)
        result = self.model.predict(x)
        result = torch.tensor(result, device=self.device)
        self.cluster_result = result
        # print('model {} fitted'.format(self.model.__class__.__name__))
        return result

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
