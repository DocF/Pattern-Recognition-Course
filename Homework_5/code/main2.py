"""
Created on Mon Apr 23 23:53:14 2018

@author: richard
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import datetime


# 设置字体
font = FontProperties(fname='/Library/Fonts/Songti.ttc')


# 生成S矩阵
data_1 = np.array([[1, 1], [2, 0], [2, 1], [0, 2], [1, 3]])
data_2 = np.array([[-1, 2], [0, 0], [-1, 0], [-1, -1], [0, -2]])

omega_1 = np.array([[1, 1, 1],  [2, 1, 1], [0, 2, 1], [1, 3, 1], [2, 0, 1]])
omega_2 = np.array([[-1, 2, 1], [0, 0, 1], [-1, 0, 1], [-1, -1, 1], [0, -2, 1]])
omega_2 = -omega_2

s = np.vstack((omega_1, omega_2))


# 初始化权重向量和参数c
w = [0, 0, 0]
c = 1
flag = 0

# 求解w向量
num_dot = 0
starttime = datetime.datetime.now()
while flag == 0:
    for i in range(0, len(s)):
        z = s[i]
        if np.dot(w, z) > 0:
            num_dot += 1
        else:
            w = w + c * z
            print(w)
            break
        if i == len(s)-1:
            flag = 1
print(num_dot)
endtime = datetime.datetime.now()
time = endtime - starttime
print(time)


# # 改进的方法求解权向量
# num_dot = 0
# starttime = datetime.datetime.now()
# while flag == 0:
#     z_sum = np.array([0, 0, 0])
#     num = 0
#     for i in range(0, len(s)):
#         z = s[i]
#         if np.dot(w, z) > 0:
#             num_dot += 1
#         else:
#             num += 1
#             z_sum = z_sum + z
#     if num == 0:
#         flag = 1
#     else:
#         z_mean = z_sum / num
#         w = w + c * z_mean
#         print(w)
# print(num_dot)
# endtime = datetime.datetime.now()
# time = endtime - starttime
# print(time)

# 画图
plt.title("改进法区分结果展示图", fontproperties=font)
x_values = np.linspace(-3, 3, 100)
y_values = [-w[0]/w[1]*x - w[2]/w[1] for x in x_values]
plt.plot(x_values, y_values, 'r')
plt.scatter(data_1[:, 0], data_1[:, 1])
plt.scatter(data_2[:, 0], data_2[:, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
# plt.savefig('../tex/img/fig7.png', dpi=800)
plt.show()
