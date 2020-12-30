# # # import torch

# # # from .config import *
# # # from .engine import get_net

# # # net = get_net(name=NET, fold=0, pretrained=False)
# # # net.load_state_dict(torch.load("D:\Kevin\Machine Learning\Cassava Leaf Disease Classification\src\models\gluon_resnet18_v1b\cldc-net=gluon_resnet18_v1b-fold=0-epoch=001-val_loss_epoch=1.6823.ckpt")["state_dict"])

# # # print(net)

# # # from joblib import Parallel, delayed
# # # from math import sqrt
# # # from time import sleep

# # # x = 10

# # # def f(i):
# # #     print(f"Called {i}")
# # #     sleep((x - i))
# # #     print(f"Finished {i}")
# # #     return i

# # # print(Parallel(n_jobs=10)(delayed(f)(i) for i in range(x)))

import timm
import pprint

# pp = pprint.PrettyPrinter(indent=4)
# list = timm.list_models()

# pp.pprint(list)

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

# model = timm.create_model("vit_base_patch16_224")
# model = timm.create_model("seresnext50_32x4d")

# count_parameters(model)
# print(model)

# batch = torch.ones(4, 512, 512)
# print(model(batch).size())

# import numpy as np

# cutmix_params = {}
# cutmix_params['alpha'] = 1
# print(np.clip(np.random.beta(
#     cutmix_params['alpha'], cutmix_params['alpha']), 0.6, 0.7))

# import torch

# from .engine import get_net
# from .config import *

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# net = get_net(name=NET, pretrained=PRETRAINED).to(device)
# net.load_state_dict(torch.load(f'D:\Kevin\Machine Learning\Cassava Leaf Disease Classification\src\models\weights\\tf_efficientnet_b4_ns\\tf_efficientnet_b4_ns_fold_0_0'))

# print(net)