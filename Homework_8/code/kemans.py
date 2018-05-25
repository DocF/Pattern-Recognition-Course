# -*-coding:utf-8-*-
# Author:Richard Fang

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data = np.loadtxt('../testSet.txt')
num_clusters = 3
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
# init = np.array([[4, 0],
#                  [-4, 0]], np.float64)
clf = KMeans(n_clusters=num_clusters,  tol=1e-4, n_init=10, verbose=1)
# clf = KMeans(n_clusters=num_clusters,   n_init=1, verbose=1)
clf.fit(data)


# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(data[:, 0], data[:, 1], 'k.', markersize=6)
# Plot the centroids as a white X
centroids = clf.cluster_centers_
print(clf.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
plt.show()



# for i in range(num_clusters):
#     x = data[clf.labels_ == i]
#     plt.scatter(x[:, 0], x[:, 1])
#
# centroids = clf.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='red', zorder=10)
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()
# print(clf.labels_)
# print(clf.cluster_centers_)
# print(clf.inertia_)
