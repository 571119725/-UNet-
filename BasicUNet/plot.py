import matplotlib.pyplot as plt
import numpy as np
import torch

data_1 = torch.load('/root/autodl-tmp/code/2.basic_unet_3/results/plot_loss_0.92.pth')
data_2 = torch.load('/root/autodl-tmp/code/2.basic_unet_3/results/plot_dice_0.92.pth')
# 生成 x 和 y1、y2 数据
# x = np.linspace(0, 10, 100)
x = data_1[0]

y1 = data_1[1]
y2 = data_2[1]
# y1 = np.sin(x)
# y2 = np.exp(x)
print(f'loss: {y1[len(y1) - 1]:.4f}, dice: {y2[len(y2) - 1]:.4f}')
# 绘制 y1 曲线和左侧 y 轴
fig, ax1 = plt.subplots()
ax1.plot(x, y1, 'b-')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Iteration Average Loss', color='b')
ax1.tick_params('y', colors='b')

# 创建新的 y 轴并将其放置在右侧
ax2 = ax1.twinx()

# 绘制 y2 曲线和右侧 y 轴
ax2.plot(x, y2, 'r-')
ax2.set_ylabel('Val Mean Dice', color='r')
ax2.tick_params('y', colors='r')

plt.savefig('/root/autodl-tmp/code/2.basic_unet_3/pictures/dice_and_loss.svg', format='svg', bbox_inches='tight')
# 显示图形
plt.show()