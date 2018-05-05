# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:06:08 2018

@author: richard
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import datetime


# ---------------------原始数据处理--------------------------------
# 设置字体
font = FontProperties(fname='/Library/Fonts/Songti.ttc')

# 读取trainData.txt数据
data = np.loadtxt('../trainData.txt')
# print(data)
# label1 = np.array([0, 0])
# label2 = np.array([0, 0])
# for i in range(0,200):
#     data_label = data[i, 2]
#     if data_label == 1.0:
#         label1 = np.row_stack((label1, data[i, 0:2]))
#     else:
#         label2 = np.row_stack((label2, data[i, 0:2]))
#
# # label1和label2分类数据
# data_label1 = label1[1:, :]
# data_label2 = label2[1:, :]
#
# print(data_label1.shape)
# print(data_label2.shape)
#
# plt.scatter(data_label1[:, 0], data_label1[:, 1])
# plt.scatter(data_label2[:, 0], data_label2[:, 1])
# plt.xlabel("x")
# plt.ylabel("y")
# plt.tight_layout()
# plt.title("训练样本分布", fontproperties=font)
# # plt.savefig('../tex/img/fig1.png', dpi=800)
# plt.show()


# ---------------------KNN算法--------------------------------
x = data[:, 0:2]
y = data[:, 2]
y -= 1
# print(y)

# 拆分训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=20180502)
# print(x_train.shape)
# print(y_test.shape)

# 训练KNN分类器

clf = neighbors.KNeighborsClassifier(n_neighbors=5)
# clf = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='kd_tree')

starttime = datetime.datetime.now()
clf.fit(x_train, y_train)
endtime = datetime.datetime.now()
time = endtime - starttime
print(time)

# 测试结果的打印
answer = clf.predict(x_test)
# print(answer)


# 准确率与召回率
precision, recall, thresholds = precision_recall_curve(y_test, answer)
probalility = clf.predict_proba(x_test)[:, 1]
# print(probalility)
# print(classification_report(y_test, answer, target_names=['1', '2']))


# 确认训练集的边界
h = .01
x_min, x_max = -6, 6
y_min, y_max = -6, 6
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 将整个测试空间的分类结果用不同颜色区分开
answer = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
z = answer.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
plt.contourf(xx, yy, z, 1, cmap=cmap_light, alpha=0.5)

# 绘制训练样本
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.Paired, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title("2-Class classification (k = 5, weights = 'distance')")
plt.tight_layout()
# plt.savefig('../tex/img/fig10.png', dpi=800)
plt.show()