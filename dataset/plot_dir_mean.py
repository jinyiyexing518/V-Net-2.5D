import matplotlib.pyplot as plt
import numpy as np

# 分别存放所有点的横坐标和纵坐标，一一对应
# y_list = np.load("./mean_array/mean1.npy")
y_list = np.load("./mean_array_without_background/mean1.npy")
x_list = np.arange(len(y_list))
y_list = list(y_list)

# 创建图并命名
plt.figure('Mean Line fig')
ax = plt.gca()

# 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
# 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度

plt.plot(x_list, y_list, color='red', linestyle='--', marker='d', linewidth=0.5)
plt.xlabel('Dir')
plt.ylabel('Mean')
plt.title("Dir Mean Curve")

plt.grid()  # 生成网格
plt.show()
