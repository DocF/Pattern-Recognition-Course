""""
Created on Mon Apr 24 22:33:54 2018

@author: richard
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import cm


# 设置字体
font = FontProperties(fname='/Library/Fonts/Songti.ttc')

data_1 = np.array([[1, 2], [2, 3], [2, 1], [3, 2]])
data_2 = np.array([[1, 1], [1, 3], [2, 2], [3, 1], [3, 3]])


# 2D分类
# ---------------双曲函数分割-------------------
x = np.linspace(-1, 5, 100)
y = [np.sqrt((e-2)*(e-2)+0.64)+2 for e in x]
plt.plot(x, y, 'r')
x = np.linspace(-1, 5, 100)
y = [-np.sqrt((e-2)*(e-2)+0.64)+2 for e in x]
plt.plot(x, y, 'r')

x = np.linspace(2.800001, 5, 100)
y = [np.sqrt((e-2)*(e-2)-0.64)+2 for e in x]
plt.plot(x, y, 'r')

x = np.linspace(2.800001, 5, 100)
y = [-np.sqrt((e-2)*(e-2)-0.64)+2 for e in x]
plt.plot(x, y, 'r')


x = np.linspace(-1.0, 1.2, 100)
y = [np.sqrt((e-2)*(e-2)-0.64)+2 for e in x]
plt.plot(x, y, 'r')

x = np.linspace(-1.0, 1.2, 100)
y = [-np.sqrt((e-2)*(e-2)-0.64)+2 for e in x]
plt.plot(x, y, 'r')

plt.scatter(data_1[:, 0], data_1[:, 1])
plt.scatter(data_2[:, 0], data_2[:, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.tight_layout()
plt.title("双曲函数区分结果展示图", fontproperties=font)
# plt.savefig('../tex/img/fig2.png', dpi=800)
plt.show()

# -----------------圆环分割---------------------
r = 0.8
a, b = (2, 2)
theta = np.arange(0, 2*np.pi, 0.01)
x = a + r * np.cos(theta)
y = b + r * np.sin(theta)
fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot(x, y, 'r')

r = 1.2
a, b = (2, 2)
theta = np.arange(0, 2*np.pi, 0.01)
x_1 = a + r * np.cos(theta)
y_1 = b + r * np.sin(theta)
axes = fig.add_subplot(111)
axes.plot(x_1, y_1, 'r')

plt.axis("equal")

plt.scatter(data_1[:, 0], data_1[:, 1])
plt.scatter(data_2[:, 0], data_2[:, 1])
plt.xlabel("x")
plt.ylabel("y")


plt.title("圆环区分结果展示图", fontproperties=font)
# plt.savefig('../tex/img/fig5.png', dpi=800)
plt.show()


# 3D分类
# ---------------正弦函数分割-------------------
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.linspace(0, 4, 100)
Y = np.linspace(0, 4, 100)
X, Y = np.meshgrid(X, Y)
Z = np.cos(np.pi*(X-2)*(Y-2))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
data_11 = np.array([[1, 2, 0], [2, 3, 0], [2, 1, 0], [3, 2, 0]])
data_12 = np.array([[1, 1, 0], [1, 3, 0], [2, 2, 0], [3, 1, 0], [3, 3, 0]])
ax.scatter(data_11[:, 0], data_11[:, 1], data_11[:, 2], marker='x')
ax.scatter(data_12[:, 0], data_12[:, 1], data_12[:, 2], marker='x')
plt.tight_layout()
plt.title("sin函数区分结果展示图", fontproperties=font)
# plt.savefig('../tex/img/fig3.png', dpi=800)
plt.show()