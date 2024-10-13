import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建数据
x = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3])
y = np.array([89, 90, 91, 92])
z = np.outer(x, y)

# 绘制三维柱状图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置颜色映射
colors = ['blue', 'cyan', 'green', 'yellow']
norm = plt.Normalize(z.min(), z.max())
scalar_map = cm.ScalarMappable(norm=norm, cmap=plt.cm.jet)
colors = scalar_map.to_rgba(z).tolist()

# 绘制每个柱子
for i in range(len(y)):
    for j in range(len(x)):
        ax.bar(j, i+1, zs=z[j][i], zdir='y', color=colors[j][i], alpha=0.8)

# 设置坐标轴标签
ax.set_xlabel('v')
ax.set_ylabel(r'$\gamma$')
ax.set_zlabel('ACC (%)')

plt.show()
