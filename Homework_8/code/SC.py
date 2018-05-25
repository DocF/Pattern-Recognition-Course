# -*-coding:utf-8-*-
# Author:Richard Fang

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import spectral_clustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle

# 产生随机数据的中心
# centers = np.array([[-4.822, 4.607],
#                  [-0.7188, -2.493],
#                  [4.377, 4.864]], np.float64)
# centers = np.array([[-3.594, 2.857],
#                  [-0.6595, 3.111],
#                  [3.998, 2.519]], np.float64)
# centers = np.array([[-0.7188, -2.493],
#                  [0.8458, -3.59],
#                  [1.149, 3.345]], np.float64)
centers = np.array([[-3.276, 1.577],
                 [3.275, 2.958],
                 [4.377, 4.864]], np.float64)

data_te = np.loadtxt('../testSet.txt')
# 变换成矩阵，输入必须是对称矩阵
metrics_metrix = (-1 * metrics.pairwise.pairwise_distances(data_te)).astype(np.int32)
metrics_metrix += -1 * metrics_metrix.min()
# 设置谱聚类函数
num_clusters_ = 3
lables = spectral_clustering(metrics_metrix, n_clusters=num_clusters_)

# 绘图
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(num_clusters_), colors):
    # 根据lables中的值是否等于k，重新组成一个True、False的数组
    my_members = lables == k
    # X[my_members, 0] 取出my_members对应位置为True的值的横坐标
    plt.plot(data_te[my_members, 0], data_te[my_members, 1], col + '.')

# plt.title('clusters=3,')
plt.show()
