import numpy as np
import matplotlib.pyplot as plt


def cal(x, y):
    z = np.cos(np.pi*(x**2-y**2-8*y+16)/4)
    return z


data_1 = np.array([[1, 2], [2, 3], [2, 1], [3, 2]])
data_2 = np.array([[1, 1], [1, 3], [2, 2], [3, 1], [3, 3]])


y_1 = np.zeros(4)
for i in range(4):
    y_1[i] = cal(data_1[i, 0], data_1[i, 1])
    print(y_1[i])

y_2 = np.zeros(5)
for i in range(5):
    y_2[i] = cal(data_2[i, 0], data_2[i, 1])
    print(y_2[i])