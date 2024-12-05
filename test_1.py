import matplotlib.pyplot as plt
import numpy as np

# 生成第一组散点图的数据
x1 = np.random.randn(100)
y1 = np.random.randn(100)

# 生成第二组散点图的数据
x2 = np.random.randn(100) + 3
y2 = np.random.randn(100) + 3

# 绘制第一组散点图
plt.scatter(x1, y1, c='r', marker='o', label='Group 1')

# 绘制第二组散点图，注意是在同一个图中继续绘制
plt.scatter(x2, y2, c='b', marker='s', label='Group 2')

# 添加标题
plt.title("Two Scatter Plots in One Figure")

# 添加 x 轴标签
plt.xlabel("X Axis")

# 添加 y 轴标签
plt.ylabel("Y Axis")

# 添加图例，用于区分不同组的散点图
plt.legend()

# 显示图形
plt.show()