# # # # import torch

# # # # from .config import *
# # # # from .engine import get_net

# # # # net = get_net(name=NET, fold=0, pretrained=False)
# # # # net.load_state_dict(torch.load("D:\Kevin\Machine Learning\Cassava Leaf Disease Classification\src\models\gluon_resnet18_v1b\cldc-net=gluon_resnet18_v1b-fold=0-epoch=001-val_loss_epoch=1.6823.ckpt")["state_dict"])

# # # # print(net)

# # # # from joblib import Parallel, delayed
# # # # from math import sqrt
# # # # from time import sleep

# # # # x = 10

# # # # def f(i):
# # # #     print(f"Called {i}")
# # # #     sleep((x - i))
# # # #     print(f"Finished {i}")
# # # #     return i

# # # # print(Parallel(n_jobs=10)(delayed(f)(i) for i in range(x)))

# import timm
# import pprint
# import torch
# import torch.nn as nn
# from vision_transformer_pytorch import VisionTransformer

# # pp = pprint.PrettyPrinter(indent=4)
# # list = timm.list_models()

# # pp.pprint(list)

# # from prettytable import PrettyTable

# # def count_parameters(model):
# #     table = PrettyTable(["Modules", "Parameters"])
# #     total_params = 0
# #     for name, parameter in model.named_parameters():
# #         if not parameter.requires_grad: continue
# #         param = parameter.numel()
# #         table.add_row([name, param])
# #         total_params+=param
# #     # print(table)
# #     print(f"Total Trainable Params: {total_params}")
# #     return total_params

# # class Identity(nn.Module):
# #     def __init__(self, _modules=None, _forward_pre_hooks=None, _forward_hooks=None, _backward_hooks=None):
# #         self._modules = _modules
# #         self._forward_pre_hooks = _forward_pre_hooks
# #         self._forward_hooks = _forward_hooks
# #         self._backward_hooks = _backward_hooks
# #         return

# #     def forward(self, x):
# #         return x

# # class Identity(nn.Module):
# #     def __init__(self):
# #         super().__init__()

# #     def forward(self, x):
# #         return x

# model = VisionTransformer.from_name('ViT-B_16')
# # model.norm = Identity(_modules={}, _forward_pre_hooks={}, _forward_hooks={}, _backward_hooks={})
# # model.head = Identity(_modules={}, _forward_pre_hooks={}, _forward_hooks={}, _backward_hooks={})
# # model.head = nn.Linear()
# model = nn.Sequential(*list(model.children()))
# # model = timm.create_model("seresnext50_32x4d")
# # model = nn.Sequential(*list(
# #             timm.create_model("vit_base_patch16_224").children()))
# # avg_pool = nn.AdaptiveAvgPool2d((1, 1))

# # count_parameters(model)
# print(model)

# batch = torch.ones(4, 3, 224, 224)
# # batch = avg_pool(batch)
# out = model(batch)
# print(out.size())

# # import numpy as np

# # cutmix_params = {}
# # cutmix_params['alpha'] = 1
# # print(np.clip(np.random.beta(
# #     cutmix_params['alpha'], cutmix_params['alpha']), 0.6, 0.7))

# # import torch

# # from .engine import get_net
# # from .config import *

# # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# # net = get_net(name=NET, pretrained=PRETRAINED).to(device)
# # net.load_state_dict(torch.load(f'D:\Kevin\Machine Learning\Cassava Leaf Disease Classification\src\models\weights\\tf_efficientnet_b4_ns\\tf_efficientnet_b4_ns_fold_0_0'))

# # print(net)

for i in range(1, 5):
    print(i)