import time
from working.utils import get_accuracy
from tqdm import tqdm
import numpy as np

import torch
from torch import optim
from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief

from .config import *
from .models.models import *


def train_one_epoch(fold, epoch, model, loss_fn, optimizer, train_loader, device, scaler, scheduler=None, schd_batch_update=False):
    model.train()

    t = time.time()
    running_loss = None
    running_accuracy = None
    if USE_TPU:
        import torch_xla.distributed.parallel_loader as pl
        train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
    total_steps = len(train_loader)

    pbar = tqdm(enumerate(train_loader), total=total_steps)
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        #print(image_labels.shape, exam_label.shape)
        if (not USE_TPU) and MIXED_PRECISION_TRAIN:
            with torch.cuda.amp.autocast():
                image_preds = model(imgs)
                loss = loss_fn(image_preds, image_labels)
                accuracy = get_accuracy(
                    image_preds.detach().cpu(), image_labels.detach().cpu())

            scaler.scale(loss).backward()
            running_loss = loss.item() if running_loss is None else (
                running_loss * .99 + loss.item() * .01)
            running_accuracy = loss.item() if running_accuracy is None else (
                running_accuracy * .99 + accuracy * .01)

            if ((step + 1) % ACCUMULATE_ITERATION == 0) or ((step + 1) == total_steps):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step()

        else:
            image_preds = model(imgs)
            loss = loss_fn(image_preds, image_labels)
            accuracy = get_accuracy(image_preds.detach().cpu(
            ).numpy(), image_labels.detach().cpu().numpy())

            loss.backward()
            running_loss = loss.item() if running_loss is None else (
                running_loss * .99 + loss.item() * .01)
            running_accuracy = loss.item() if running_accuracy is None else (
                running_accuracy * .99 + accuracy * .01)

            if ((step + 1) % ACCUMULATE_ITERATION == 0) or ((step + 1) == total_steps):
                if USE_TPU:
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step()

        if ((LEARNING_VERBOSE and (step + 1) % VERBOSE_STEP == 0)) or ((step + 1) == total_steps):
            description = f'[{fold}/{FOLDS - 1}][{epoch}/{MAX_EPOCHS - 1}][{step + 1}/{total_steps}] Loss: {running_loss:.4f} | Accuracy: {running_accuracy:.4f}'
            pbar.set_description(description)

        # break

    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(fold, epoch, model, loss_fn, valid_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    if USE_TPU:
        import torch_xla.distributed.parallel_loader as pl
        valid_loader = pl.ParallelLoader(valid_loader, [device]).per_device_loader(device)

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)
        image_preds_all += [torch.argmax(image_preds,
                                         1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item()*image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((LEARNING_VERBOSE and (step + 1) % VERBOSE_STEP == 0)) or ((step + 1) == len(valid_loader)):
            description = f'[{fold}/{FOLDS - 1}][{epoch}/{MAX_EPOCHS - 1}] Validation Loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)

        # break

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print(f'[{fold}/{FOLDS - 1}][{epoch}/{MAX_EPOCHS - 1}] Validation Multi-Class Accuracy = {(image_preds_all == image_targets_all).mean():.4f}')

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()


def get_net(name, pretrained=False):
    if name not in nets.keys():
        net = GeneralizedCassavaClassifier(name, pretrained=pretrained)
    else:
        net = nets[name](pretrained=pretrained)

    return net


def get_optimizer_and_scheduler(net):
    # Optimizers
    if OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(
            params=net.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5
        )
    elif OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(
            net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    elif OPTIMIZER == "AdaBelief":
        optimizer = AdaBelief(net.parameters(
        ), lr=LEARNING_RATE, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
    elif OPTIMIZER == "RangerAdaBelief":
        optimizer = RangerAdaBelief(
            net.parameters(), lr=LEARNING_RATE, eps=1e-12, betas=(0.9, 0.999), print_change_log=False)
    else:
        optimizer = optim.SGD(
            net.parameters(), lr=LEARNING_RATE)

    # Schedulers
    if SCHEDULER == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=0,
            factor=0.1,
            verbose=LEARNING_VERBOSE
        )
    elif SCHEDULER == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=5, eta_min=0)
    elif SCHEDULER == "OneCycleLR":
        steps_per_epoch = len(self.train_dataloader())
        # steps_per_epoch = len(self.train_dataloader())//self.trainer.accumulate_grad_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            pct_start=0.1,
            div_factor=1e3,
            max_lr=1e-2,
            steps_per_epoch=steps_per_epoch,
            epochs=MAX_EPOCHS
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}
    else:
        scheduler = None

    return optimizer, scheduler

def get_device():
    if not USE_GPU and USE_TPU:
        return torch.device('cpu')
    elif USE_TPU:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    elif USE_GPU:
        return torch.device('cuda:0')