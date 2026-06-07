# ============================================================
# datasets.py  —  DIV2K 训练集 + Kodak 测试集
# ============================================================
import glob
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from config import (
    DIV2K_TRAIN_HR, DIV2K_VAL_HR, KODAK_DIR,
    CROP_SIZE, BATCH_SIZE, SEED
)


# ── DIV2K ─────────────────────────────────────────────────────

class DIV2KDataset(Dataset):
    """
    DIV2K 高分辨率训练集。
    每张图随机 crop CROP_SIZE×CROP_SIZE，随机水平/垂直翻转，随机 90° 旋转。
    """
    def __init__(self, root=DIV2K_TRAIN_HR, crop_size=CROP_SIZE, augment=True):
        self.paths = sorted(
            glob.glob(f'{root}/*.png') +
            glob.glob(f'{root}/*.jpg')
        )
        if not self.paths:
            raise FileNotFoundError(
                f"DIV2K 图片未找到，请检查路径：{root}\n"
                "下载命令：\n"
                "  wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip\n"
                "  unzip DIV2K_train_HR.zip -d ./data/DIV2K/"
            )
        self.crop_size = crop_size
        self.augment   = augment
        print(f'DIV2K dataset: {len(self.paths)} images from {root}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')

        # 随机 crop
        i, j, h, w = T.RandomCrop.get_params(img, (self.crop_size, self.crop_size))
        img = TF.crop(img, i, j, h, w)

        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
            if random.random() > 0.5:
                img = TF.vflip(img)
            k = random.randint(0, 3)
            if k > 0:
                img = TF.rotate(img, 90 * k)

        return TF.to_tensor(img), 0   # 无标签


class DIV2KValDataset(Dataset):
    """DIV2K 验证集，center crop，用于 epoch 末快速检验。"""
    def __init__(self, root=DIV2K_VAL_HR, crop_size=CROP_SIZE):
        self.paths = sorted(
            glob.glob(f'{root}/*.png') +
            glob.glob(f'{root}/*.jpg')
        )
        self.tf = T.Compose([
            T.CenterCrop(crop_size),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.tf(img), 0


def get_div2k_loaders(batch_size=BATCH_SIZE, num_workers=4):
    train_ds = DIV2KDataset(augment=True)
    val_ds   = DIV2KValDataset()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


# ── Kodak ─────────────────────────────────────────────────────

class KodakDataset(Dataset):
    """Kodak-24，整图推理，无标签。"""
    def __init__(self, root=KODAK_DIR):
        self.paths = sorted(
            glob.glob(f'{root}/*.png') +
            glob.glob(f'{root}/*.jpg')
        )
        if not self.paths:
            raise FileNotFoundError(
                f"Kodak 图片未找到，请检查路径：{root}\n"
                "下载命令（需要翻墙或手动下载）：\n"
                "  mkdir -p ./data/kodak\n"
                "  # 从 http://r0k.us/graphics/kodak/ 下载 24 张 PNG"
            )
        print(f'Kodak dataset: {len(self.paths)} images from {root}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return TF.to_tensor(img), 0


def get_kodak_loader(num_workers=2):
    # batch_size=1，因为 Kodak 图像尺寸不统一
    ds = KodakDataset()
    return DataLoader(ds, batch_size=1, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
