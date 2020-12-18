import os

import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .config import *
from .utils import *
from .transforms import *


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


class CassavaTrain(Dataset):
    def __init__(
            self, data, data_root=TRAIN_IMAGES_DIR, transforms=None) -> None:

        self.data = data[:SUBSET_SIZE] if USE_SUBSET else data
        self.data_root = data_root
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        path = os.path.join(self.data_root, self.data[index, 0])
        img = get_img(path)

        if self.transforms:
            img = self.transforms(image=img)['image']

        # do label smoothing
        target = self.data[index, 1]
        return img, target


class CassavaTest(Dataset):
    def __init__(
            self, data, data_root=TRAIN_IMAGES_DIR, transforms=None) -> None:

        self.data = data[:SUBSET_SIZE] if USE_SUBSET else data
        self.data_root = data_root
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        path = os.path.join(self.data_root, self.data[index, 0])
        img = get_img(path)

        if self.transforms:
            img = self.transforms(image=img)['image']

        # do label smoothing
        return img


def get_train_dataloader(train):
    return DataLoader(
        CassavaTrain(
            data=train,
            transforms=get_train_transforms()),
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=CPU_WORKERS,
        shuffle=True)


def get_valid_dataloader(valid):
    return DataLoader(
        CassavaTrain(
            data=valid,
            transforms=get_valid_transforms()),
        batch_size=VALID_BATCH_SIZE,
        num_workers=CPU_WORKERS,
        shuffle=False)
