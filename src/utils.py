import os
import sys
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
        print(config.WEIGHTS_PATH)
        os.mkdir(config.WEIGHTS_PATH)
        print(f"Created Folder {config.WEIGHTS_PATH}")
    except:
        print("Error:", sys.exc_info()[0])
    try:
        os.mkdir(os.path.join(config.WEIGHTS_PATH, f'{config.NET}'))
        print(f"Created Folder {os.path.join(config.WEIGHTS_PATH, f'{config.NET}')}")
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")

class AverageLossMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.curr_batch_avg_loss = 0
        self.avg = 0
        self.running_total_loss = 0
        self.count = 0

    def update(self, curr_batch_avg_loss: float, batch_size: str):
        self.curr_batch_avg_loss = curr_batch_avg_loss
        self.running_total_loss += curr_batch_avg_loss * batch_size
        self.count += batch_size
        self.avg = self.running_total_loss / self.count

class AccuracyMeter:
    def __init__(self):        
        self.reset()
        
    def reset(self):
        self.score = 0
        self.count = 0
        self.sum = 0

    def update(self, y_pred, y_true, batch_size=1):
        # so we just need to count total num of images / batch_size
        #self.count += num_steps
        self.batch_size = batch_size
        self.count += self.batch_size
        # this part here already got an acc score for the 4 images, so no need divide batch size
        self.score = accuracy_score(y_true, y_pred)
        total_score = self.score * self.batch_size
        self.sum += total_score
        
    @property
    def avg(self):        
        self.avg_score = self.sum/self.count
        return self.avg_score
