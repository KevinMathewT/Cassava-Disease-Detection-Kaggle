# # # -*- coding: utf-8 -*-
# # import torch
# # import math

# # dtype = torch.float
# # device = torch.device("cpu")
# # # device = torch.device("cuda:0")  # Uncomment this to run on GPU

# # # Create Tensors to hold input and outputs.
# # # By default, requires_grad=False, which indicates that we do not need to
# # # compute gradients with respect to these Tensors during the backward pass.
# # x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
# # y = torch.sin(x)

# # # Create random Tensors for weights. For a third order polynomial, we need
# # # 4 weights: y = a + b x + c x^2 + d x^3
# # # Setting requires_grad=True indicates that we want to compute gradients with
# # # respect to these Tensors during the backward pass.
# # a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
# # b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
# # c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
# # d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

# # learning_rate = 1e-6
# # for t in range(2000):
# #     # Forward pass: compute predicted y using operations on Tensors.
# #     y_pred = a + b * x + c * x ** 2 + d * x ** 3

# #     # Compute and print loss using operations on Tensors.
# #     # Now loss is a Tensor of shape (1,)
# #     # loss.item() gets the scalar value held in the loss.
# #     loss = (y_pred - y).pow(2).sum()
# #     if t % 100 == 99:
# #         print(t, loss.item())

# #     # Use autograd to compute the backward pass. This call will compute the
# #     # gradient of loss with respect to all Tensors with requires_grad=True.
# #     # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
# #     # the gradient of the loss with respect to a, b, c, d respectively.
# #     loss.backward()

# #     # Manually update weights using gradient descent. Wrap in torch.no_grad()
# #     # because weights have requires_grad=True, but we don't need to track this
# #     # in autograd.
# #     with torch.no_grad():
# #         a -= learning_rate * a.grad
# #         b -= learning_rate * b.grad
# #         c -= learning_rate * c.grad
# #         d -= learning_rate * d.grad

# #         # Manually zero the gradients after updating weights
# #         a.grad = None
# #         b.grad = None
# #         c.grad = None
# #         d.grad = None

# # print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

# # from src.config import *
# # from src.utils import *

# # print(NET)
# # config.NET = "kevin"
# # print(NET)
# # create_dirs()
from prettytable import PrettyTable

import torch
import torch.nn as nn

from .engine import get_net

import timm
from vision_transformer_pytorch import VisionTransformer
import pprint

# def l2_norm(input, axis=1):
#     norm = torch.norm(input, 2, axis, True)
#     output = torch.div(input, norm)
#     return output


# class BinaryHead(nn.Module):
#     def __init__(self, num_class=5, emb_size=2048, s=16.0):
#         super(BinaryHead, self).__init__()
#         self.s = s
#         self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

#     def forward(self, fea):
#         fea = l2_norm(fea)
#         logit = self.fc(fea) * self.s
#         return logit

# # net = timm.create_model("vit_base_patch16_224", pretrained=False)
# # net.norm = nn.Identity()
# # net.head = nn.Identity()
# # print(net)
# N_CLASSES = 5
# # x = torch.ones((2, 3, 224, 224))
# # print(net(x).shape)

# class ViTBase16_BH(nn.Module):
#     name = "ViTBase16_BH"

#     def __init__(self, pretrained=False):
#         super().__init__()
#         self.net = timm.create_model("vit_base_patch16_224", pretrained=False)
#         self.net.norm = nn.Identity()
#         self.net.head = nn.Identity()
#         self.fea_bn = nn.BatchNorm1d(768)
#         self.fea_bn.bias.requires_grad_(False)
#         self.binary_head = BinaryHead(N_CLASSES, emb_size=768, s=1)
#         self.dropout = nn.Dropout(p=0.2)

#     def forward(self, x):
#         x = self.net(x)
#         x = self.fea_bn(x)
#         # fea = self.dropout(fea)
#         x = self.binary_head(x)
#         return x


# net = ViTBase16_BH(pretrained=False)
# x = torch.ones((2, 3, 224, 224))
# print(net(x).shape)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    # print(table)
    print(f"Total Trainable Params: {total_params / 1000000:.1f}M")
    return total_params

print(timm.list_models("vit*"))

# net = get_net("vit_small_patch16_224")
# count_parameters(net)

# import time
# import random
# import numpy as np

# avg = 0.0

# for j in range(100):
#     start = time.time()
#     for i in range(1000):
#         # do = np.random.uniform(0., 1., size=1)[0] > 0.5
#         do = np.random.random() > 0.5
#     avg += time.time() - start

# print(f"Time: {avg / 100.0}")