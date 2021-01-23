import torch
import torch.nn as nn

import timm


# class GeneralizedCassavaClassifier(nn.Module):
#     def __init__(self, model_arch, n_class=5, pretrained=False):
#         super().__init__()
#         self.name = model_arch
#         self.model = timm.create_model(model_arch, pretrained=pretrained)
#         model_list = list(self.model.children())
#         model_list[-1] = nn.Linear(
#             in_features=model_list[-1].in_features,
#             out_features=n_class,
#             bias=True
#         )
#         self.model = nn.Sequential(*model_list)

#     def forward(self, x):
#         x = self.model(x)
#         return x


# net = GeneralizedCassavaClassifier(model_arch="resnext50_32x4d")

# for module in net.modules():
#     if isinstance(module, nn.BatchNorm2d):
#         if hasattr(module, 'weight'):
#             print(module.weight.requires_grad)
#         if hasattr(module, 'bias'):
#             print(module.bias.requires_grad)

# for module in net.modules():
#     if isinstance(module, nn.BatchNorm2d):
#         if hasattr(module, 'weight'):
#             module.weight.requires_grad_(False)
#         if hasattr(module, 'bias'):
#             module.bias.requires_grad_(False)
#         module.eval()

# for module in net.modules():
#     if isinstance(module, nn.BatchNorm2d):
#         if hasattr(module, 'weight'):
#             print(module.weight.requires_grad)
#         if hasattr(module, 'bias'):
#             print(module.bias.requires_grad)

# net.train()

# for module in net.modules():
#     if isinstance(module, nn.BatchNorm2d):
#         if hasattr(module, 'weight'):
#             print(module.weight.requires_grad)
#         if hasattr(module, 'bias'):
#             print(module.bias.requires_grad)

# print(timm.list_models())
net = timm.create_model("tf_efficientnet_b4_ns", pretrained=False)
print(net)