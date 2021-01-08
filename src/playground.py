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

if USE_TPU:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl

os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"

df = pd.read_csv(TRAIN_FOLDS)
dataloader = get_infer_dataloader(infer=df)
device = get_device(n=0)
net = get_net(name=NET, pretrained=False)
net.load_state_dict(torch.load("../input/model-weights/SEResNeXt50_32x4d_BH_fold_2_11.bin", map_location=torch.device('cpu')))
net = xmp.MpModelWrapper(net) if USE_TPU else net
net = net.to(device)

preds = np.empty((0, 5), dtype=np.float64)
for images, labels in tqdm(dataloader):
    images, labels = images.to(device), labels.to(device)
    predictions = net(images).detach().cpu().numpy()
    preds = np.concatenate([preds, predictions], axis=0)
    break
    
print(preds.shape)
ids = df["image_id"].to_numpy().reshape(-1, 1)
preds = np.concatenate([ids, preds], axis=1)
print(preds.shape)
preds = pd.DataFrame(preds, cols=['id', '0', '1', '2', '3', '4'])