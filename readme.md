+ **任务**：使用MONAI框架中的UNet网络完成KiTS19数据集的肾脏和肿瘤分割任务。

+ MONAI框架中提供了UNet、UNetr等网络的实现，只需要直接引入网络进行训练就可以。

> MONAI源码中提供的网络有这些：https://github.com/Project-MONAI/MONAI/tree/7baf2822c4cee81b54585974adde4a51d8040536/monai/networks/nets

+ **代码参考**：https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/unetr_btcv_segmentation_3d.ipynb，源码相关论文见参考文献*《UNETR Transformers for 3D Medical Image Segmentation》*

> 代码注释见本仓库UNet文件夹下的unetr.py文件
> 代码环境：**python3.8.10 + pytorch1.11 + monai框架**

+ **数据集**：使用KiTS19数据集

> 官方数据描述[Data - Grand Challenge (grand-challenge.org)](https://kits19.grand-challenge.org/data/)
> 论文描述见参考文献*《The KiTS19 Challenge Data》*
> 数据集下载链接：https://pan.baidu.com/s/1AOQDjPz9ye32DH-oDS0WDw   提取码：d7jk （如果觉得下载太慢找我用U盘拷）
> CT影像查看可以使用软件Slicer

+ 学习资料：

  深度学习资料：[《动手学深度学习》 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh.d2l.ai/)

  pytorch：见参考文献中的*《Pytorch官方教程》*