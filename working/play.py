# import torch

# from .config import *
# from .engine import get_net

# net = get_net(name=NET, fold=0, pretrained=False)
# net.load_state_dict(torch.load("D:\Kevin\Machine Learning\Cassava Leaf Disease Classification\working\models\gluon_resnet18_v1b\cldc-net=gluon_resnet18_v1b-fold=0-epoch=001-val_loss_epoch=1.6823.ckpt")["state_dict"])

# print(net)

from joblib import Parallel, delayed
from math import sqrt
from time import sleep

x = 10

def f(i):
    print(f"Called {i}")
    sleep((x - i))
    print(f"Finished {i}")
    return i

print(Parallel(n_jobs=10)(delayed(f)(i) for i in range(x)))
