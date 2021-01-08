import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.dataset import CassavaDataset, get_infer_dataloader
from src.engine import get_device, get_net, get_optimizer_and_scheduler, train_one_epoch, valid_one_epoch
from src.config import *
from src.utils import *
from src.loss import FocalCosineLoss, SmoothCrossEntropyLoss, bi_tempered_logistic_loss

# from IPython.display import FileLinks
# FileLinks(WEIGHTS_PATH)

# df = pd.read_csv(TRAIN_FOLDS)
# print("Here")
# dataloader = get_infer_dataloader(infer=df)
# print("Here")
# device = get_device(n=0)
# print("Here")
net = get_net(name=NET, pretrained=False)
print(net)
# print("Here")
# net = net.to(device)
# print("Here")
# if USE_TPU:
#     import torch_xla.utils.serialization as xser
#     net.load_state_dict(xser.load("../input/model-weights/SEResNeXt50_32x4d_BH_fold_2_11.bin"))
# else:
net.load_state_dict(torch.load("../input/model-weights/SEResNeXt50_32x4d_BH_fold_2_11.bin"))
# print("Here")

# preds = np.empty((0, 5), dtype=np.float64)
# for images, labels in tqdm(dataloader):
#     images, labels = images.to(device), labels.to(device)
#     predictions = net(images).cpu().numpy()
#     preds = np.concatenate([preds, predictions], axis=0)
    
# print(preds.shape)