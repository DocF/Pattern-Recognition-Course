# -*- coding: utf-8 -*-
"""
Created on Wed May  3 22:25:07 2018

@author: richard
"""
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


w1 = np.array([[2, 0], [2, 2], [2, 4], [3, 3]])
w2 = np.array([[0, 3], [-2, 2], [-1, -1], [1, -2], [3, -1]])

w3 = np.array([[1, 1], [2, 0], [2, 1], [0, 2], [1, 3]])
w4 = np.array([[-1, 2], [0, 0], [-1, 0], [-1, -1], [0, -2]])


w, mean1, mean2 = fisher(w1, w2)  # 调用函数，得到参数w,
print(mean1, mean2)
w0 = -0.5*(np.dot(w.T, mean1) + np.dot(w.T, mean2))
print(w)
print(w0)

plt.scatter(w1[:, 0], w1[:, 1], c='#99CC99')
plt.scatter(w2[:, 0], w2[:, 1], c='#FFCC00')
line_x = np.arange(min(np.min(w1[:, 0]), np.min(w2[:, 0])),
                   max(np.max(w1[:, 0]), np.max(w2[:, 0])),
                   step=1)

line_y = (- w[0] * line_x - w0) / w[1]
plt.plot(line_x, line_y)
plt.arrow(-2.5, -1.640625, 6.5, 4.265625, width=0.03)
plt.axis("equal")
plt.xlabel('x')
plt.ylabel('y')
plt.title("Fisher classifier(w1,w2)")
plt.tight_layout()
plt.savefig('../tex/img/fig12.png', dpi=800)
plt.show()

print(-w[0]/w[1])
