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

def plot_data_and_crop(val_inputs, val_labels, slice_num):
    pict_img = np.squeeze(val_inputs.cpu().numpy())
    pict_label = np.squeeze(val_labels.cpu().numpy())
    for i in range(pict_label.shape[0]):
        for j in range(pict_label.shape[1]):
            if pict_label[i, j, slice_num] == 2:
                pict_label[i, j, slice_num] = 1
    mask_img = np.multiply(pict_img[:, :, slice_num], pict_label[:, :, slice_num])

    plt.figure("check", (16, 8))
    plt.subplot(1, 3, 1)
    plt.title("(a)",font={'size':20}, y=-0.1)
    plt.imshow(np.squeeze(val_labels.cpu().numpy())[:, :, slice_num])
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("(b)",font={'size':20}, y=-0.1)
    plt.imshow(mask_img, cmap="gray")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("(c)",font={'size':20}, y=-0.1)
    plt.imshow(pict_img[:, :, slice_num], cmap="gray")
    plt.axis('off')
    plt.savefig('/root/autodl-tmp/code/2.basic_unet_3/pictures/best_cut_and_crop.svg', format='svg', bbox_inches='tight')
    plt.show()

case_name = "case_00053"
directory = '/root/autodl-tmp/code/2.basic_unet_3/results'
root_dir = tempfile.mkdtemp() if directory is None else directory
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    epoch_iterator_val = tqdm(val_loader, desc="Testing")
    for (i, batch) in enumerate(epoch_iterator_val):
        img_name = os.path.split(val_ds[i]["image"].meta["filename_or_obj"])[0].split("/")[5]
        if img_name == case_name:    
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (64, 64, 64), 4, model)            
            plot_data_and_crop(val_inputs, torch.argmax(val_outputs, dim = 1), 40)