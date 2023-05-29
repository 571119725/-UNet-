import os
import shutil
import tempfile
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    Rand3DElasticd,
    ResizeWithPadOrCropd,
    AdjustContrastd
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import BasicUnet

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
# 设置数据路径
directory = '/root/autodl-tmp/code/2.basic_unet_3/results'
root_dir = tempfile.mkdtemp() if directory is None else directory
# compose将transforms组合在一起
#图像尺寸为[1, 259, 223, 74]
train_transforms = Compose(
    [
        #加载图片的值和元数据，参数keys是data_dicts中设置的keys，表示对image还是label做变换
        LoadImaged(keys=["image", "label"], image_only = False),
        #自动添加一个通道的维度，保证通道在第一维度
        EnsureChannelFirstd(keys=["image", "label"]),
        #对图像进行一个方向变换，转为RAS坐标
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        #对图像进行重采样，体素间距重采样为[1.5, 1.5, 2.0]
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.62, 1.62, 3.2),
            mode=("bilinear"),
        ),
        #对图像值强度进行归一化，由a的范围归一到b，，不在a范围的值设置为0
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-79,
            a_max=304,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        #根据key所指定的大于0的部分，对于图像的有效部分进行裁剪
        CropForegroundd(keys=["image", "label"], source_key="label", margin = 20),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(160, 96, 64), mode="constant"),
        # AdjustContrastd(keys=["image"], gamma = 2),
        #将图像裁剪为4个子图
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(64, 64, 64),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        #随机旋转，按着0轴旋转，旋转概率为0.1
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.5,
        ),
        #随机旋转，按着1轴旋转，旋转概率为0.1
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.5,
        ),
        #随机旋转，按着2轴旋转，旋转概率为0.1
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.5,
        ),
        #随机旋转，概率为0.1，旋转次数为3
        RandRotate90d(
            keys=["image", "label"],
            prob=0.5,
            max_k=4,
        ),
        #随机强度转换，强度偏移量为[-0.1, 0.1]，概率为0.5
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.40,
            prob=0.50,
        ),
    ]
)

val_transforms = Compose(
    [
        #加载图片的值和元数据，参数keys是data_dicts中设置的keys，表示对image还是label做变换
        LoadImaged(keys=["image", "label"], image_only = False),
        #自动添加一个通道的维度，保证通道在第一维度
        EnsureChannelFirstd(keys=["image", "label"]),
        #对图像进行一个方向变换，转为RAS坐标
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        #对图像进行重采样，体素间距重采样为[1.5, 1.5, 2.0]
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.62, 1.62, 3.2),
            mode=("bilinear"),
        ),
        #对图像值强度进行归一化，由a的范围归一到b，，不在a范围的值设置为0
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-79,
            a_max=304,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        #根据key所指定的大于0的部分，对于图像的有效部分进行裁剪
        CropForegroundd(keys=["image", "label"], source_key="label", margin = 20),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(160, 96, 64), mode="constant"),
    ]
)
#数据
data_dir = "/root/autodl-tmp/code/data/"
split_json = "dataset_2.json"
datasets = data_dir + split_json
#从json文件中加载数据集（数据集路径，是否用于分割任务，字典键值）
#训练集
datalist = load_decathlon_datalist(datasets, is_segmentation = True, data_list_key = "training")
val_files = load_decathlon_datalist(datasets, is_segmentation= True, data_list_key= "validation")
#具有缓存机制的dataset，在每一个epoch训练之前，把训练的数据加载进缓存
#读取图像数并进行图像转换
#（将image和label的地址或值存为字典，transform，要缓存的项目数，，缓存数据占总数的百分比默认1，要使用的工作进程数）
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num= 170,
    cache_rate= 1.0,
    num_workers = 14,
)
val_ds = CacheDataset(
    data = val_files,
    transform = val_transforms,
    cache_num = 40,
    cache_rate = 1.0,
    num_workers = 14,
)
#加载图像，（加载的数据集，batch_size，是否打乱，使用的进程数，是否将数据保存在pin_memory区）
train_loader = DataLoader(train_ds, batch_size = 4, shuffle=True, num_workers=14, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size = 4, shuffle = False, num_workers = 14, pin_memory = True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#设置网络
model = BasicUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    dropout=0.0
).to(device)
#损失函数
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
#为每个卷积层寻找适合的卷积实现算法，加速网络训练
torch.backends.cudnn.benchmark = True
#设置网络参数更新方式
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.35, verbose=1, min_lr = 1e-6, patience=5)
def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (64, 64, 64), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_labels_tensor) for val_labels_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_outputs_convert = [post_pred(val_outputs_tensor) for val_outputs_tensor in val_outputs_list]
            dice_metric(y_pred = val_outputs_convert, y = val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, max_iterations))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val

#def train(0, train_loader, 0.0, 0)
def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    #返回一个迭代器，并会不断打印迭代进度条，desc为进度条前缀，允许调整窗口大小
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1        
        #总训练次数加一
        global_step += 1
        #读取图像和标签并放入GPU
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        #将图像放入模型进行训练
        logit_map = model(x)
        # print("\nx'size:" + str(x.shape) + "out'shape:" + str(logit_map.shape) + "y'shape:" + str(y.shape) + "\n")
        #计算训练损失
        loss = loss_function(logit_map, y)
        #向后传播
        loss.backward()
        #.item()返回指定位置的高精度值
        #计算损失和
        epoch_loss += loss.item()
        #更新参数
        optimizer.step()
        #更新参数之后清除梯度
        optimizer.zero_grad()
        #设置进度条前缀，为当前训练次数，训练总数，本次训练损失
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))
        if(global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            scheduler.step(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print(f'Saved! Current best average dice:{dice_val_best}')
            else:
                print(f'Not saved! Current best average dice:{dice_val_best}, current average dice:{dice_val}')
    #返回总训练次数，最佳Dice系数，最佳训练次数
    return global_step, dice_val_best, global_step_best

def plot_loss_and_metric():
    plt.figure("train",(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plot_data_1 = [x, y]
    torch.save(plot_data_1, '/root/autodl-tmp/code/2.basic_unet_3/results/plot_loss.pth')
    plt.xlabel("Iteration")
    plt.plot(plot_data_1[0], plot_data_1[1])
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [eval_num * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plot_data_2 = [x, y]
    torch.save(plot_data_2, '/root/autodl-tmp/code/2.basic_unet_3/results/plot_dice.pth')
    plt.xlabel("Iteration")
    plt.plot(plot_data_2[0],plot_data_2[1])
    plt.show()

max_iterations = 34000
# max_iterations = 3400
eval_num = 170
#将输入的张量转换为离散值，采用3位独热编码
post_label = AsDiscrete(to_onehot=3)
#采用argmax函数
post_pred = AsDiscrete(argmax=True, to_onehot=3)
#计算两个张量之间的平均Dice系数
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
#全局训练次数
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_3_0.8936.pth")))
#当训练iteration小于max_iterations，继续训练
if __name__ == '__main__':
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
    plot_loss_and_metric()