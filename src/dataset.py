import os
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

# from fmix import sample_mask, make_low_freq_image, binarise_mask

from .config import *
from .utils import *
from .transforms import *

if USE_TPU:
    import torch_xla.core.xla_model as xm


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
        #lam, mask = sample_mask(**self.fmix_params)

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

        #assert self.output_label==True and self.one_hot_label==True

        # mix target
        rate = mask.sum() / H / W
        target = rate * target + (1. - rate) * labels[fmix_ix]
        #print(target, mask, img)
        #assert False


def cutmix(img, target, labels, df, transforms, cutmix_params, data_root):
    #print(img.sum(), img.shape)
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

    #print('-', img.sum())
    # print(target)
    #assert False


class MonocularDepthImageMasking:
    def __init__(self):
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        self.midas.cuda().eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.midas_transforms.default_transform

    def get_depth(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to("cuda")
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
            ).squeeze()
        output = prediction.cpu().numpy()
        img_min = np.min(output)
        img_max = np.max(output)
        return output > ((img_min+img_max)/3)

    def crop_image(self, image, depth):
        depth = depth.astype(int)
        mask_3d = np.stack((depth, depth, depth), axis=2)
        masked_arr = np.where(mask_3d == 1, image, mask_3d).astype(np.uint8)
        c = np.where(masked_arr != [0, 0, 0])
        x_max = max(c[1])
        x_min = min(c[1])
        y_max = max(c[0])
        y_min = min(c[0])
        image = masked_arr[y_min:y_max, x_min:x_max]
        return image

    def __call__(self, image):
        process = image
        depth = self.get_depth(process)
        image = self.crop_image(image, depth)
        return image


class CassavaDataset(Dataset):
    def __init__(self, df, data_root,
                 transforms=None,
                 output_label=True,
                 one_hot_label=False,
                 fmix_params={
                     'alpha': 1.,
                     'decay_power': 3.,
                     'shape': (H, W),
                     'max_soft': True,
                     'reformulate': False
                 },
                 cutmix_params={
                     'alpha': 1,
                 }
                 ):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.fmix_params = fmix_params
        self.cutmix_params = cutmix_params

        self.output_label = output_label
        self.one_hot_label = one_hot_label

        if DO_DEPTH_MASKING:
            self.masker = MonocularDepthImageMasking()

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

        if DO_DEPTH_MASKING:
            img = self.masker(img)

        if self.transforms:
            img = self.transforms(image=img)['image']

        if DO_FMIX and np.random.uniform(0., 1., size=1)[0] > 0.5:
            img, target = fmix(img, target, self.labels, self.df,
                               self.transforms, self.fmix_params, self.data_root)

        if DO_CUTMIX and np.random.uniform(0., 1., size=1)[0] > 0.5:
            img, target = fmix(img, target, self.labels, self.df,
                               self.transforms, self.fmix_params, self.data_root)

        # do label smoothing
        #print(type(img), type(target))
        if self.output_label == True:
            return img, target
        else:
            return img


def get_train_dataloader(train, data_root=TRAIN_IMAGES_DIR):
    if USE_TPU:
        train_dataset = CassavaDataset(train, data_root, transforms=get_train_transforms(
        ), output_label=True, one_hot_label=False)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),  # divide dataset among this many replicas
            rank=xm.get_ordinal(),  # which replica/device/core
            shuffle=True)
        return DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            sampler=train_sampler,
            num_workers=CPU_WORKERS,
            drop_last=True)
    else:
        return DataLoader(
            CassavaDataset(train, data_root, transforms=get_train_transforms(
            ), output_label=True, one_hot_label=False),
            batch_size=TRAIN_BATCH_SIZE,
            drop_last=True,
            num_workers=CPU_WORKERS,
            shuffle=True)


def get_valid_dataloader(valid, data_root=TRAIN_IMAGES_DIR):
    if USE_TPU:
        valid_dataset = CassavaDataset(valid, data_root, transforms=get_valid_transforms(
        ), output_label=True, one_hot_label=False)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False)
        return DataLoader(
            valid_dataset,
            batch_size=VALID_BATCH_SIZE,
            sampler=valid_sampler,
            num_workers=CPU_WORKERS,
            drop_last=False)
    else:
        return DataLoader(
            CassavaDataset(valid, data_root, transforms=get_valid_transforms(
            ), output_label=True, one_hot_label=False),
            batch_size=VALID_BATCH_SIZE,
            drop_last=True,
            num_workers=CPU_WORKERS,
            shuffle=False)


def get_infer_dataloader(infer, data_root=TRAIN_IMAGES_DIR):
    OK = False
    if USE_TPU and OK:
        infer_dataset = CassavaDataset(infer, data_root, transforms=get_valid_transforms(
        ), output_label=True, one_hot_label=False)
        infer_sampler = torch.utils.data.distributed.DistributedSampler(
            infer_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False)
        return DataLoader(
            infer_dataset,
            batch_size=32,
            sampler=infer_sampler,
            num_workers=CPU_WORKERS,
            drop_last=False)
    else:
        return DataLoader(
            CassavaDataset(infer, data_root, transforms=get_valid_transforms(
            ), output_label=True, one_hot_label=False),
            batch_size=32,
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
