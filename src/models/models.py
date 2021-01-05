import torch
import torch.nn as nn
from vision_transformer_pytorch import VisionTransformer

import timm
from ..config import *


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class BinaryHead(nn.Module):
    def __init__(self, num_class=N_CLASSES, emb_size=2048, s=16.0):
        super(BinaryHead, self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea) * self.s
        return logit


class SEResNeXt50_32x4d_BH(nn.Module):
    name = "SEResNeXt50_32x4d_BH"

    def __init__(self, pretrained=False):
        super().__init__()
        self.model_arch = "seresnext50_32x4d"
        self.net = nn.Sequential(*list(
            timm.create_model(self.model_arch, pretrained=pretrained).children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(N_CLASSES, emb_size=2048, s=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        img_feature = self.net(x)
        img_feature = self.avg_pool(img_feature)
        img_feature = img_feature.view(img_feature.size(0), -1)
        fea = self.fea_bn(img_feature)
        # fea = self.dropout(fea)
        output = self.binary_head(fea)

        return output


class ViTBase16_BH(nn.Module):
    name = "ViTBase16_BH"

    def __init__(self, pretrained=False):
        super().__init__()
        # self.model_arch = "vit_base_patch16_224"
        # self.net = timm.create_model("vit_base_patch16_224")
        self.net = VisionTransformer.from_name('ViT-B_16', num_classes=5)
        self.net.head = nn.Linear(in_features=768, out_features=768, bias=True)
        self.fea_bn = nn.BatchNorm1d(768)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(N_CLASSES, emb_size=768, s=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        img_feature = self.net(x)
        fea = self.fea_bn(img_feature)
        # fea = self.dropout(fea)
        output = self.binary_head(fea)

        return output


class GeneralizedCassavaClassifier(nn.Module):
    def __init__(self, model_arch, n_class=N_CLASSES, pretrained=False):
        super().__init__()
        self.name = model_arch
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        model_list = list(self.m.children())
        model_list[-1] = nn.Linear(
            in_features=model_list[-1].in_features,
            out_features=n_class,
            bias=True
        )
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        x = self.model(x)
        return x


nets = {
    "SEResNeXt50_32x4d_BH": SEResNeXt50_32x4d_BH,
    "ViTBase16_BH": ViTBase16_BH,
}
