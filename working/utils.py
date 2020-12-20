import os
import random
import numpy as np
import torch
import cv2
from sklearn.metrics import accuracy_score
from torch.nn.functional import normalize

from torch.utils.data import DataLoader, Dataset

from .config import *
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
    im_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    return im_rgb

def get_accuracy(predictions, targets, normalize=True):
    predictions = torch.argmax(predictions, dim=1)
    return accuracy_score(targets, predictions, normalize=normalize)

def create_dirs():
    try:
        os.mkdir(f'./working/models/weights/')
        os.mkdir(f'./working/models/weights/{NET}/')
    except:
        return
