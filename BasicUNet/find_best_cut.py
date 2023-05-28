import os
import shutil
import tempfile
import torch
import matplotlib.pyplot as plt
import pylab
from tqdm import tqdm
import nibabel as nib
import numpy as np
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
    ResizeWithPadOrCropd,
    RandRotate90d,
    AdjustContrastd,
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
data_dir = "/root/autodl-tmp/code/data/"
split_json = "dataset_2.json"
datasets = data_dir + split_json
val_files = load_decathlon_datalist(datasets, is_segmentation = True, data_list_key="validation")
val_ds = CacheDataset(
    data = val_files,
    transform = val_transforms,
    cache_num = 40,
    cache_rate = 1.0,
    num_workers = 12
)
val_loader = DataLoader(
    val_ds, 
    batch_size = 1, 
    shuffle = False, 
    num_workers = 12, 
    pin_memory = True
)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = BasicUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    dropout=0.0
).to(device)

def plot_data_and_label(val_inputs, val_labels, val_outputs):
    plt.figure("check", (12, 10))
    plt.subplot(2, 3, 1)
    plt.title("image")
    plt.axis('off')
    plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_num_1], cmap="gray")
    plt.subplot(2, 3, 2)
    plt.title("label")
    plt.axis('off')
    plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_num_1])
    plt.subplot(2, 3, 3)
    plt.title("output")
    plt.axis('off')
    plt.imshow(torch.argmax(val_outputs, dim = 1).detach().cpu()[0, :, :, slice_num_1])
    plt.subplot(2, 3, 4)
    plt.title("image")
    plt.axis('off')
    plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_num_2], cmap="gray")
    plt.subplot(2, 3, 5)
    plt.title("label")
    plt.axis('off')
    plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_num_2])
    plt.subplot(2, 3, 6)
    plt.title("output")
    plt.axis('off')
    plt.imshow(torch.argmax(val_outputs, dim = 1).detach().cpu()[0, :, :, slice_num_2])
    plt.savefig('/root/autodl-tmp/code/2.basic_unet_3/pictures/best_cut_pict.svg', format='svg', bbox_inches='tight')
    plt.show()

case_num = 0
directory = '/root/autodl-tmp/code/2.basic_unet_3/results'
root_dir = tempfile.mkdtemp() if directory is None else directory
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
slice_num_1 = 31
slice_num_2 = 40
best_dice = 0.0
file_name = ""
post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
with torch.no_grad():
    epoch_iterator_val = tqdm(val_loader, desc="Testing")
    for (i, batch) in enumerate(epoch_iterator_val):
        img_name = os.path.split(val_ds[i]["image"].meta["filename_or_obj"])[0].split("/")[5]
        val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
        val_outputs = sliding_window_inference(val_inputs, (64, 64, 64), 4, model)
        
        val_labels_list = decollate_batch(val_labels)
        val_labels_convert = [post_label(val_labels_tensor) for val_labels_tensor in val_labels_list]
        val_outputs_list = decollate_batch(val_outputs)
        val_outputs_convert = [post_pred(val_outputs_tensor) for val_outputs_tensor in val_outputs_list]
        
        dice_metric(y_pred = val_outputs_convert, y = val_labels_convert)
        tmp = dice_metric.aggregate().item()
        print(f'name: {img_name},dice: {tmp}')
        if tmp > best_dice:
            best_dice = tmp
            file_name = img_name
        dice_metric.reset()
    print(f'best crop file name: {file_name},best dice: {best_dice}')
    for (i, batch) in enumerate(epoch_iterator_val):
        img_name = os.path.split(val_ds[i]["image"].meta["filename_or_obj"])[0].split("/")[5]
        if img_name == file_name:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (64, 64, 64), 4, model)
            plot_data_and_label(val_inputs, val_labels, val_outputs)
    