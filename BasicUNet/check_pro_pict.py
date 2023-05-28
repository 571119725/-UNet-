import os
import shutil
import tempfile
import torch
import matplotlib.pyplot as plt
import pylab
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
    GaussianSharpen,
    Rand3DElasticd,
    ResizeWithPadOrCropd,
    AdjustContrastd
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import Unet

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

def plot_data_and_label(val_inputs, val_labels, slice_num_1, slice_num_2):
    plt.figure("check", (12, 10))
    plt.subplot(2, 2, 1)
    plt.title("image")
    plt.axis('off')
    plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_num_1], cmap="gray")
    plt.subplot(2, 2, 2)
    plt.title("label")
    plt.axis('off')
    plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_num_1])
    plt.subplot(2, 2, 3)
    plt.title("image")
    plt.axis('off')
    plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_num_2], cmap="gray")
    plt.subplot(2, 2, 4)
    plt.title("label")
    plt.axis('off')
    plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_num_2])
    plt.savefig('/root/autodl-tmp/code/2.basic_unet_3/pictures/image_and_label.svg', format='svg', bbox_inches='tight')
    plt.show()

case_name = "case_00053"
with torch.no_grad():
    epoch_iterator_val = tqdm(val_loader, desc="Testing")
    for (i, batch) in enumerate(epoch_iterator_val):
        img_name = os.path.split(val_ds[i]["image"].meta["filename_or_obj"])[0].split("/")[5]
        if img_name == case_name:    
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())         
            plot_data_and_label(val_inputs, val_labels, 30, 80)