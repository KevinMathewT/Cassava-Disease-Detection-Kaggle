import os
import random
import numpy as np
import torch
import cv2

from torch.utils.data import DataLoader, Dataset

from .config import *
from .dataset import CassavaTrain
from .transforms import *

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

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