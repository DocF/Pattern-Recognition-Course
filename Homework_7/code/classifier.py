#-*-coding:utf-8-*-
"""
Created on Wed May  9 00:10:53 2018

@author: Richard Fang
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def cal_cov_and_avg(samples):
    """
    给定一个类别的数据，计算协方差矩阵和平均向量
    :param samples:
    :return:
    """
    u1 = np.mean(samples, axis=0)
    cov_m = np.zeros((samples.shape[1], samples.shape[1]))
    for s in samples:
        t = s - u1
        cov_m += t * t.reshape(2, 1)
    return cov_m, u1


def fisher(c_1, c_2):
    """
    fisher算法实现
    :param c_1:类型1数据
    :param c_2:类型2数据
    :return: w*，即最优投影方向
    """
    cov_1, u1 = cal_cov_and_avg(c_1)
    cov_2, u2 = cal_cov_and_avg(c_2)
    s_w = cov_1 + cov_2
    u, s, v = np.linalg.svd(s_w)  # 奇异值分解
    s_w_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)
    return (np.dot(s_w_inv, u1 - u2),u1 ,u2)


# data
data1 = np.array([[3, 4], [3, 8], [2, 6], [4, 6]])
data2 = np.array([[3, 0], [3, -4], [1, -2], [5, -2]])
omega1 = data1.T
omega2 = data2.T

# mean of omega1 and omega2
mu1 = np.mean(omega1, axis=1)
mu2 = np.mean(omega2, axis=1)

# calculate the expect of covariance
cov1 = np.cov(omega1)*3/4
cov2 = np.cov(omega2)*3/4

# calculate the det and inv
det1 = np.linalg.det(cov1)
det2 = np.linalg.det(cov2)
inv1 = np.linalg.inv(cov1)
inv2 = np.linalg.inv(cov2)

x = np.linspace(-5, 11, 1000)
y = [3/16*(i-3)**2 - math.log(4)/8 + 2 for i in x]


print(mu1)
print(mu2)
print(cov1)
print(cov2)
print(det1)
print(det2)
print(inv1)
print(inv2)



plt.plot(x,y)


# plt.scatter(data1[:, 0], data1[:, 1])
# plt.scatter(data2[:, 0], data2[:, 1])
# plt.scatter(mu1[0], mu1[1])
# plt.scatter(mu2[0], mu2[1])
#
# a1 = 2
# b1 = 2
# theta = np.arange(0, 2*np.pi, np.pi/100)
# x1 = 3 + a1 * np.sin(theta)
# y1 = -2 + b1 * np.cos(theta)
# plt.plot(x1, y1)
#
# a2 = 1
# b2 = 2
# theta = np.arange(0, 2*np.pi, np.pi/100)
# x2 = 3 + a2 * np.sin(theta)
# y2 = 6 + b2 * np.cos(theta)
# plt.plot(x2, y2)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.text(mu1[0]+1, mu1[1]+2, '$R_1$', ha='center', va='bottom', fontsize=9)
# plt.text(mu2[0]+1.5, mu2[1]+2, '$R_2$', ha='center', va='bottom', fontsize=9)
# plt.savefig('../tex/img/fig1.png', dpi=800)

sampleNo = 1000000
rd_mu1 = np.array(mu1)
Sigma1 = np.array(cov1)
R1 = np.linalg.cholesky(Sigma1)
s1 = np.dot(np.random.randn(sampleNo, 2), R1) + rd_mu1
# y1 = s1[:, 1] - np.array([3/16*(i-3)**2 - math.log(4)/8 + 2 for i in s1[:, 0]])
y3 = s1[:, 1] - 2
# idx1 = np.where(y1<0)
idx3 = np.where(y3<0)
# print(idx1[0].shape)
print(idx3[0].shape)
# plt.plot(s1[:, 0], s1[:, 1], 'bo', alpha=0.3)

rd_mu2 = np.array(mu2)
Sigma2 = np.array(cov2)
R2 = np.linalg.cholesky(Sigma2)
s2 = np.dot(np.random.randn(sampleNo, 2), R2) + rd_mu2
y4 = s2[:, 1] - 2
# y2 = s2[:, 1] - np.array([3/16*(i-3)**2 - math.log(4)/8 + 2 for i in s2[:, 0]])
# idx2 = np.where(y2 > 0)
idx4 = np.where(y4 > 0)
# print(idx2[0].shape)
print(idx4[0].shape)
# plt.plot(s2[:, 0], s2[:, 1], 'ro', alpha=0.3)


# w, mean1, mean2 = fisher(s1, s2)  # 调用函数，得到参数w,
# w0 = -0.5*(np.dot(w.T, mean1) + np.dot(w.T, mean2))
#
# line_x = np.arange(-5, 10, 1/100)
#
# line_y = (- w[0] * line_x - w0) / w[1]
# print(line_y)
# plt.plot(line_x, line_y)


plt.xlabel('x')
plt.ylabel('y')
plt.title("Classifier")
plt.axis('equal')
plt.tight_layout()
plt.show()