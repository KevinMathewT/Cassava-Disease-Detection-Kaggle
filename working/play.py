# # import torch

# # from .config import *
# # from .engine import get_net

# # net = get_net(name=NET, fold=0, pretrained=False)
# # net.load_state_dict(torch.load("D:\Kevin\Machine Learning\Cassava Leaf Disease Classification\working\models\gluon_resnet18_v1b\cldc-net=gluon_resnet18_v1b-fold=0-epoch=001-val_loss_epoch=1.6823.ckpt")["state_dict"])

# # print(net)

# # from joblib import Parallel, delayed
# # from math import sqrt
# # from time import sleep

# # x = 10

# # def f(i):
# #     print(f"Called {i}")
# #     sleep((x - i))
# #     print(f"Finished {i}")
# #     return i

# # print(Parallel(n_jobs=10)(delayed(f)(i) for i in range(x)))

# import timm
# import pprint

# # pp = pprint.PrettyPrinter(indent=4)
# # list = timm.list_models()

# # pp.pprint(list)

# from prettytable import PrettyTable

# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad: continue
#         param = parameter.numel()
#         table.add_row([name, param])
#         total_params+=param
#     # print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params

# model = timm.create_model("densenet121")
# count_parameters(model)

import numpy as np

cutmix_params = {}
cutmix_params['alpha'] = 1
print(np.clip(np.random.beta(
    cutmix_params['alpha'], cutmix_params['alpha']), 0.6, 0.7))
