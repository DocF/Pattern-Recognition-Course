# -*-coding:utf-8-*-
# Author:Richard Fang
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def cal_label(src, init):
    """
    calculate the label of data
    :param src:
    :param init:
    :return:
    """
    data = src
    cent = init
    label = np.zeros(data.shape[0], np.int32)
    for i in range(data.shape[0]):
        dis = [0]*cent.shape[0]
        # print(dis)
        for j in range(cent.shape[0]):
            d = data[i] - cent[j]
            # print(d)
            # dis[j] = d[0]**2 + d[1]**2
            dis[j] = np.linalg.norm(d, ord=2)
            # print(dis[j])
        label[i] = dis.index(min(dis))
    # print(label)
    return label


def cal_error(src, cent, label):
    """
    calculate the sum of the square of error
    :param src:
    :param cent:
    :param label:
    :return:
    """
    data = src
    data_label = label
    centroids = cent
    distance = 0
    for i in range(data.shape[0]):
        # print(data_label[i])
        dis = data[i] - centroids[data_label[i]]
        distance += norm(dis, ord=2)
    return distance


def Kmeans(src, init):
    """
    Kmeans function is used for unsupervised clustering
    :param src: input array for clustering
    :param init: initial centroids
    :return: label for data
    """
    data = src
    init_cent = init
    data_label = cal_label(data, init_cent)
    error = [0, 1]
    intera = 0
    while(np.abs(error[1]-error[0])>1e-8):
        intera += 1
        error[0] = error[1]
        for j in range(init_cent.shape[0]):
            # print(data[data_label == j])
            init_cent[j] = np.mean(data[data_label == j],axis=0)
        data_label = cal_label(data, init_cent)
        error[1] = cal_error(data, init_cent, data_label)
        # init_cent = 0 * init_cent
    return data_label, init_cent, error[1], intera


data_te = np.loadtxt('../testSet.txt')
num_clusters = 2
# init = np.array([[-4.822, 4.607],
#                  [-0.7188, -2.493],
#                  [4.377, 4.864]], np.float64)
# init = np.array([[-3.594, 2.857],
#                  [-0.6595, 3.111],
#                  [3.998, 2.519]], np.float64)
# init = np.array([[-0.7188, -2.493],
#                  [0.8458, -3.59],
#                  [1.149, 3.345]], np.float64)
# init = np.array([[-3.276, 1.577],
#                  [3.275, 2.958],
#                  [4.377, 4.864]], np.float64)

init = np.array([[0, 0],
                 [1, 1]], np.float64)
# init = np.array([[-2, -2],
#                  [-2, 2],
#                  [2, -2],
#                  [2, 2]], np.float64)

answer = Kmeans(data_te, init)
data_telabel = answer[0]

for i in range(num_clusters):
    x = data_te[data_telabel == i]
    plt.scatter(x[:, 0], x[:, 1])

centroids = answer[1]
print("三类样本均值", centroids)

sum_error = answer[2]
print("总的误差", sum_error)

itera = answer[3]
print("迭代步骤", itera)
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='red', zorder=10)
#
# plt.xlabel('x')
# plt.ylabel('y')
# # plt.title("clusters=，分类结果")
# plt.tight_layout()
# plt.savefig("../tex/img/fig6.png", dpi=800)
# plt.show()

