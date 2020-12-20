from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomBrightness,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, RandomContrast, GaussianBlur,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
import cv2

from .config import *
from .utils import *
from working import config


# def get_train_transforms():
#     # return Compose([
#     #         RandomResizedCrop(H, W),
#     #         Transpose(p=0.5),
#     #         HorizontalFlip(p=0.5),
#     #         VerticalFlip(p=0.5),
#     #         ShiftScaleRotate(p=0.5),
#     #         HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
#     #         RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
#     #         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
#     #         CoarseDropout(p=0.5),
#     #         Cutout(p=0.5),
#     #         ToTensorV2(p=1.0),
#     #     ], p=1.)
#     return Compose([
#         Resize(H, W, p=1.),
#         OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),
#         OneOf([MotionBlur(blur_limit=3), MedianBlur(
#             blur_limit=3), GaussianBlur(blur_limit=3)], p=0.5),
#         VerticalFlip(p=0.5),
#         HorizontalFlip(p=0.5),
#         ShiftScaleRotate(
#             shift_limit=0.2,
#             scale_limit=0.2,
#             rotate_limit=20,
#             interpolation=cv2.INTER_LINEAR,
#             border_mode=cv2.BORDER_REFLECT_101,
#             p=1,
#         ),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[
#                   0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
#         ToTensorV2(p=1.0),
#     ], p=1.)


# def get_valid_transforms():
#     return Compose([
#         Resize(H, W, p=1.),
#         Resize(H, W),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[
#                   0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
#         ToTensorV2(p=1.0),
#     ], p=1.)


# def get_inference_transforms():
#     return Compose([
#         Resize(H, W, p=1.),
#         Transpose(p=0.5),
#         HorizontalFlip(p=0.5),
#         VerticalFlip(p=0.5),
#         HueSaturationValue(hue_shift_limit=0.2,
#                            sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
#         RandomBrightnessContrast(
#             brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[
#                   0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
#         ToTensorV2(p=1.0),
#     ], p=1.)

def get_train_transforms():
    return Compose([
            RandomResizedCrop(config.H, config.W),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms():
    return Compose([
            CenterCrop(config.H, config.W, p=1.),
            Resize(config.H, config.W),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_inference_transforms():
    return Compose([
            RandomResizedCrop(config.H, config.W),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)