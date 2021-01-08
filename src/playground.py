import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.dataset import CassavaDataset
from src.engine import get_device, get_net, get_optimizer_and_scheduler, train_one_epoch, valid_one_epoch
from src.config import *
from src.utils import *
from src.loss import FocalCosineLoss, SmoothCrossEntropyLoss, bi_tempered_logistic_loss

from IPython.display import FileLinks
FileLinks(WEIGHTS_PATH)

df = pd.read_csv(TRAIN_FOLDS)
dataset = CassavaDataset(df=df,
                         data_root=TRAIN_IMAGES_DIR,
                         transforms=get_valid_transforms())
dataloader = DataLoader(dataset,
                        batch_size=3,
                        drop_last=False,
                        num_workers=0,
                        shuffle=False)
device = get_device(n=0)
net = get_net(name=NET, pretrained=False)
net.load_state_dict(torch.load("./generated/weights\SEResNeXt50_32x4d_BH/SEResNeXt50_32x4d_BH_fold_2_11.bin"))
net = net.to(device)

preds = np.empty((0, 5), dtype=np.float64)
for images, labels in tqdm(dataloader):
    images, labels = images.to(device), labels.to(device)
    predictions = net(images).detach().cpu().numpy()
    preds = np.concatenate([preds, predictions], axis=0)
    
print(preds.shape)