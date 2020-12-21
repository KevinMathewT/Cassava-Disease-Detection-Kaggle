import os
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

# from fmix import sample_mask, make_low_freq_image, binarise_mask

from .config import *
from .utils import *
from .transforms import *


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def fmix(img, target, labels, df, transforms, fmix_params, data_root):
    with torch.no_grad():
        # lam, mask = sample_mask(**self.fmix_params)

        lam = np.clip(np.random.beta(
            fmix_params['alpha'], fmix_params['alpha']), 0.6, 0.7)

        # Make mask, get mean / std
        mask = make_low_freq_image(
            fmix_params['decay_power'], fmix_params['shape'])
        mask = binarise_mask(
            mask, lam, fmix_params['shape'], fmix_params['max_soft'])

        fmix_ix = np.random.choice(df.index, size=1)[0]
        fmix_img = get_img(
            "{}/{}".format(data_root, df.iloc[fmix_ix]['image_id']))

        if transforms:
            fmix_img = transforms(image=fmix_img)['image']

        mask_torch = torch.from_numpy(mask)

        # mix image
        img = mask_torch * img + (1. - mask_torch) * fmix_img

        # print(mask.shape)

        # assert self.output_label==True and self.one_hot_label==True

        # mix target
        rate = mask.sum() / H / W
        target = rate * target + (1. - rate) * labels[fmix_ix]
        # print(target, mask, img)
        # assert False


def cutmix(img, target, labels, df, transforms, cutmix_params, data_root):
    # print(img.sum(), img.shape)
    with torch.no_grad():
        cmix_ix = np.random.choice(df.index, size=1)[0]
        cmix_img = get_img(
            "{}/{}".format(data_root, df.iloc[cmix_ix]['image_id']))
        if transforms:
            cmix_img = transforms(image=cmix_img)['image']

        lam = np.clip(np.random.beta(
            cutmix_params['alpha'], cutmix_params['alpha']), 0.3, 0.4)
        bbx1, bby1, bbx2, bby2 = rand_bbox((H, W), lam)

        img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

        rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
        target = rate * target + (1. - rate) * labels[cmix_ix]

    # print('-', img.sum())
    # print(target)
    # assert False


class CassavaDataset(Dataset):
    def __init__(self, df, data_root,
                 transforms=None,
                 output_label=True,
                 one_hot_label=False,
                 do_fmix=False,
                 fmix_params={
                     'alpha': 1.,
                     'decay_power': 3.,
                     'shape': (H, W),
                     'max_soft': True,
                     'reformulate': False
                 },
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 }
                 ):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params

        self.output_label = output_label
        self.one_hot_label = one_hot_label

        if output_label == True:
            self.labels = self.df['label'].values
            # print(self.labels)

            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max()+1)[self.labels]
                # print(self.labels)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.labels[index]
        else:
            target = None

        img = get_img("{}/{}".format(self.data_root,
                                     self.df.loc[index]['image_id']))

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            img, target = fmix(img, target, self.labels, self.df,
                               self.transforms, self.fmix_params, self.data_root)

        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            img, target = fmix(img, target, self.labels, self.df,
                               self.transforms, self.fmix_params, self.data_root)

        # do label smoothing
        # print(type(img), type(target))
        if self.output_label == True:
            return img, target
        else:
            return img


def get_train_dataloader(train, data_root=TRAIN_IMAGES_DIR):
    dataset = CassavaDataset(train, data_root, transforms=get_train_transforms(
    ), output_label=True, one_hot_label=False, do_fmix=False, do_cutmix=False),
    return DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        pin_memory=False,
        drop_last=False,
        num_workers=CPU_WORKERS,
        shuffle=True)


def get_valid_dataloader(valid, data_root=TRAIN_IMAGES_DIR):
    dataset = CassavaDataset(valid, data_root, transforms=get_valid_transforms(
    ), output_label=True, one_hot_label=False, do_fmix=False, do_cutmix=False),
    return DataLoader(
        dataset,
        batch_size=VALID_BATCH_SIZE,
        pin_memory=False,
        drop_last=False,
        num_workers=CPU_WORKERS,
        shuffle=False)


def get_loaders(fold):
    train_folds = pd.read_csv(TRAIN_FOLDS)
    train = train_folds[train_folds.fold != fold]
    valid = train_folds[train_folds.fold == fold]

    train_loader = get_train_dataloader(
        train.drop(['fold'], axis=1))
    valid_loader = get_valid_dataloader(
        valid.drop(['fold'], axis=1))

    return train_loader, valid_loader
