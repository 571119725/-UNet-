import os
import shutil
import tempfile
import torch
import matplotlib.pyplot as plt
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
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
# 设置数据路径
directory = './results'
root_dir = tempfile.mkdtemp() if directory is None else directory
# compose将transforms组合在一起
train_transforms = Compose(
    [
        #加载图片的值和元数据，参数keys是data_dicts中设置的keys，表示对image还是label做变换
        LoadImaged(keys=["image", "label"]),
        #自动添加一个通道的维度，保证通道在第一维度
        EnsureChannelFirstd(keys=["image", "label"]),
        #对图像进行一个方向变换，转为RAS坐标
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        #对图像进行重采样，体素间距重采样为[1.5, 1.5, 5.0]
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        #对图像值强度进行归一化，由a的范围归一到b，，不在a范围的值设置为0
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        #根据key所指定的大于0的部分，对于图像的有效部分进行裁剪
        CropForegroundd(keys=["image", "label"], source_key="image"),
        #将图像裁剪为4个子图
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
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
            prob=0.10,
        ),
        #随机旋转，按着1轴旋转，旋转概率为0.1
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        #随机旋转，按着2轴旋转，旋转概率为0.1
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        #随机旋转，概率为0.1，旋转次数为3
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        #随机强度转换，强度偏移量为[-0.1, 0.1]，概率为0.5
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)
val_transforms = Compose(
    [
        #加载图片的值和元数据，参数keys是data_dicts中设置的keys，表示对image还是label做变换
        LoadImaged(keys=["image", "label"]),
        #自动添加一个通道的维度，保证通道在第一维度
        EnsureChannelFirstd(keys=["image", "label"]),
        #对图像进行一个方向变换，转为RAS坐标
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        #对图像进行重采样，体素间距重采样为[1.5, 1.5, 5.0]
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        #对图像值强度进行归一化，由a的范围归一到b，，不在a范围的值设置为0
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        #根据key所指定的大于0的部分，对于图像的有效部分进行裁剪
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)
#数据
data_dir = "/dataset/"
split_json = "dataset_0.json"
datasets = data_dir + split_json
#从json文件中加载数据集（数据集路径，是否用于分割任务，字典键值）
#训练集
datalist = load_decathlon_datalist(datasets, True, "training")
#验证集
val_files = load_decathlon_datalist(datasets, True, "validation")
#具有缓存机制的dataset，在每一个epoch训练之前，把训练的数据加载进缓存
#读取图像数并进行图像转换
#（将image和label的地址或值存为字典，transform，要缓存的项目数，，缓存数据占总数的百分比默认1，要使用的工作进程数）
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=8,
)
#加载图像，（加载的数据集，batch_size，是否打乱，使用的进程数，是否将数据保存在pin_memory区）
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

#检查数据和可视化
slice_map = {
    "img0035.nii.gz": 170,
    "img0036.nii.gz": 230,
    "img0037.nii.gz": 204,
    "img0038.nii.gz": 204,
    "img0039.nii.gz": 204,
    "img0040.nii.gz": 180,
}
#检查数据和可视化
def data_shape(data_num = 0):
    case_num = data_num
    img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")
    plt.figure("image", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    #detach()得到的张量不会被计算梯度，.cpu()将数据从其他设备拿到cpu
    plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())
    plt.show()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#设置网络
model = UNETR(
    in_channels=1,
    out_channels=14,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)
#损失函数
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
#为每个卷积层寻找适合的卷积实现算法，加速网络训练
torch.backends.cudnn.benchmark = True
#设置网络参数更新方式
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

def validation(epoch_iterator_val):
    #将模型设置为评估模式
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            #读取验证集中的图片和标签并传入GPU
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            #滑动窗口取出图像进行训练
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            #将标签数据去除batch维度，返回图像列表
            val_labels_list = decollate_batch(val_labels)
            #将label转换为独热编码
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            #将训练输出数据去除batch维度，返回图像列表
            val_outputs_list = decollate_batch(val_outputs)
            #将结果转换为独热编码
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            #计算两个张量之间的平均Dice系数
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            #设置进度条前缀，为当前训练次数，训练总数，验证损失
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    #返回验证集平均Dice系数
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
        #读取图像和标签并放入GPU
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        #将图像放入模型进行训练
        logit_map = model(x)
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
        #训练次数每过eval_num(500)次，或训练完成
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            #返回一个迭代器，数据为验证集，并打印迭代进度条，desc为进度条前缀，允许调整窗口大小
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            #验证机平均Dice系数
            dice_val = validation(epoch_iterator_val)
            #计算平均训练损失
            epoch_loss /= step
            #记录平均训练损失
            epoch_loss_values.append(epoch_loss)
            #记录平均Dice系数
            metric_values.append(dice_val)
            #如果训练效果比记录的好
            if dice_val > dice_val_best:
                #更新最佳Dice系数
                dice_val_best = dice_val
                #更新最佳迭代次数
                global_step_best = global_step
                #保存当前模型数据，保存在results文件夹中
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
            #训练效果没有记录的好，答应最好的Dice系数和本次Dice系数
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        #总训练次数加一
        global_step += 1
    #返回总训练次数，最佳Dice系数，最佳训练次数
    return global_step, dice_val_best, global_step_best

max_iterations = 25000
eval_num = 500
#将输入的张量转换为离散值，采用14位独热编码
post_label = AsDiscrete(to_onehot=14)
#采用argmax函数
post_pred = AsDiscrete(argmax=True, to_onehot=14)
#计算两个张量之间的平均Dice系数
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
#全局训练次数
global_step = 0
#最佳Dice系数（最大为1）
dice_val_best = 0.0
#达到最佳训练效果的训练次数
global_step_best = 0
#每过eval_num次训练之后的训练损失集合
epoch_loss_values = []
#每过eval_num次训练之后的Dice系数集合
metric_values = []
#当训练iteration小于max_iterations，继续训练
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
#打印效果最好的一次训练的Dice系数以及迭代的次数
print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")

#绘制平均训练损失和平均Dice系数
def plot_loss_and_metric():
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [eval_num * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.show()


#检查效果最好的模型的输入和输出以及label
def check_best_model(best_num = 4):
    case_num = best_num
    #加载效果最好的模型数据
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
    #设置为评估模式
    model.eval()
    with torch.no_grad():
        img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        #在图像后面插入一个维度
        val_inputs = torch.unsqueeze(img, 1).cuda()
        val_labels = torch.unsqueeze(label, 1).cuda()
        val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1) 
        plt.title("image")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("label")
        plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
        plt.subplot(1, 3, 3)
        plt.title("output")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]])
        plt.show()