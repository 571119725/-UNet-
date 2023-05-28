#### 任务及建议

+ **任务**：使用MONAI框架中的UNet网络完成KiTS19数据集的肾脏和肿瘤分割任务。

+ MONAI框架中提供了UNet、UNetr等网络的实现，只需要直接引入网络进行训练就可以。

  > MONAI源码中提供的网络有这些：https://github.com/Project-MONAI/MONAI/tree/7baf2822c4cee81b54585974adde4a51d8040536/monai/networks/nets
  >
  > 建议使用其中的BasicUNet网络，参数少，结构简单，效果好

+ **代码参考**：https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/unetr_btcv_segmentation_3d.ipynb

  > MONAI源码中提供的网络有这些：https://github.com/Project-MONAI/MONAI/tree/7baf2822c4cee81b54585974adde4a51d8040536/monai/networks/nets
  >
  > 建议使用其中的BasicUNet网络，参数少，结构简单，效果好

+ **数据集**：使用KiTS19数据集

  > 官方数据描述[Data - Grand Challenge (grand-challenge.org)](https://kits19.grand-challenge.org/data/)
  >
  > 论文描述见参考文献*《The KiTS19 Challenge Data》*
  >
  > 数据集下载链接：https://pan.baidu.com/s/1AOQDjPz9ye32DH-oDS0WDw   提取码：d7jk （如果觉得下载太慢找我用U盘拷）
  >
  > CT影像查看可以使用软件Slicer

+ **学习资料**：

  深度学习资料：[《动手学深度学习》 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh.d2l.ai/)

  pytorch：见参考文献中的*《Pytorch官方教程》*



#### 文件描述

+ **UNet**：原始代码和注释

+ **BasicUNet**：basicUNet代码和相关文件

  > 1. basicunet.py：网络训练代码
  > 2. check_best_crop.py：使用分割效果最好的图像，展示分割后器官，并保存图像
  > 3. check_pro_pict.py：查看未分割的图像和标签，展示并保存
  > 4. find_best_cut.py：找到分割效果最好的图像，展示并保存
  > 5. plot.py：通过保存的plot_data重新绘制分割准确度和loss值增长图像并保存
  > 6. read_dir.py：读取数据目录并保存为json文件
  > 7. result_analysis.py：计算测试集中的多器官分割准确度
  > 8. dataset：源数据
  > 9. pictures：图像存储目录
  > 10. results：训练后模型参数存储目录