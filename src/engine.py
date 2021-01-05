import time
from src.utils import get_accuracy
from sklearn.metrics import accuracy_score
import numpy as np

import torch
from torch import optim
from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief

from .config import *
from .models.models import *
from .utils import AccuracyMeter, AverageLossMeter

if USE_TPU:
    import torch_xla.core.xla_model as xm


def train_one_epoch(fold, epoch, model, loss_fn, optimizer, train_loader, device, scaler, scheduler=None, schd_batch_update=False):
    model.train()

    print_fn = print if not USE_TPU else xm.master_print
    t = time.time()
    running_loss = AverageLossMeter()
    running_accuracy = AccuracyMeter()
    total_steps = len(train_loader)
    pbar = enumerate(train_loader)

    for step, (imgs, image_labels) in pbar:
        if not USE_TPU:
            imgs = imgs.to(device).float()
            image_labels = image_labels.to(device).long()
        if USE_TPU:
            imgs = imgs.to(device, dtype=torch.float32)
            image_labels = image_labels.to(device, dtype=torch.int64)
        curr_batch_size = imgs.size(0)

        # print(image_labels.shape, exam_label.shape)
        if (not USE_TPU) and MIXED_PRECISION_TRAIN:
            with torch.cuda.amp.autocast():
                image_preds = model(imgs)
                loss = loss_fn(image_preds, image_labels)
                running_loss.update(
                    curr_batch_avg_loss=loss.item(), batch_size=curr_batch_size)
                running_accuracy.update(
                    y_pred=image_preds.detach().cpu(),
                    y_true=image_labels.detach().cpu(),
                    batch_size=curr_batch_size)

            scaler.scale(loss).backward()

            if ((step + 1) % ACCUMULATE_ITERATION == 0) or ((step + 1) == total_steps):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step()

        else:
            image_preds = model(imgs)
            loss = loss_fn(image_preds, image_labels)
            loss.backward()

            if ((step + 1) % ACCUMULATE_ITERATION == 0) or ((step + 1) == total_steps):
                if USE_TPU:
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step()
            
            running_loss.update(
                curr_batch_avg_loss=loss.item(), batch_size=curr_batch_size)
            running_accuracy.update(
                y_pred=image_preds.detach().cpu(),
                y_true=image_labels.detach().cpu(),
                batch_size=curr_batch_size)

        if ((LEARNING_VERBOSE and (step + 1) % VERBOSE_STEP == 0)) or ((step + 1) == total_steps):
            description = f'[{fold}/{FOLDS - 1}][{epoch}/{MAX_EPOCHS - 1}][{step + 1}/{total_steps}] Loss: {running_loss.avg:.4f} | Accuracy: {running_accuracy.avg:.4f}'
            print_fn(description)

        # break

    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(fold, epoch, model, loss_fn, valid_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    print_fn = print if not USE_TPU else xm.master_print
    t = time.time()
    running_loss = AverageLossMeter()
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    total_steps = len(valid_loader)
    pbar = enumerate(valid_loader)

    for step, (imgs, image_labels) in pbar:
        if not USE_TPU:
            imgs = imgs.to(device).float()
            image_labels = image_labels.to(device).long()
        if USE_TPU:
            imgs = imgs.to(device, dtype=torch.float32)
            image_labels = image_labels.to(device, dtype=torch.int64)

        image_preds = model(imgs)
        image_preds_all += [torch.argmax(image_preds,
                                         1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        running_loss.update(curr_batch_avg_loss=loss.item(),
                            batch_size=image_labels.shape[0])
        sample_num += image_labels.shape[0]

        if ((LEARNING_VERBOSE and (step + 1) % VERBOSE_STEP == 0)) or ((step + 1) == len(valid_loader)):
            description = f'[{fold}/{FOLDS - 1}][{epoch}/{MAX_EPOCHS - 1}] Validation Loss: {running_loss.avg:.4f}'
            print_fn(description)

        # break

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print_fn(f'[{fold}/{FOLDS - 1}][{epoch}/{MAX_EPOCHS - 1}] Validation Multi-Class Accuracy = {accuracy_score(image_targets_all, image_preds_all):.4f}')

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(running_loss.avg)
        else:
            scheduler.step()


def get_net(name, pretrained=False):
    if name not in nets.keys():
        net = GeneralizedCassavaClassifier(name, pretrained=pretrained)
    else:
        net = nets[name](pretrained=pretrained)

    if USE_TPU:
        import torch_xla.distributed.xla_multiprocessing as xmp
        net = xmp.MpModelWrapper(net)
    return net


def get_optimizer_and_scheduler(net, dataloader):
    # Optimizers
    if OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(
            params=net.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5
        )
    elif OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(
            net.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
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
            verbose=LEARNING_VERBOSE)
    elif SCHEDULER == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=5, eta_min=0)
    elif SCHEDULER == "OneCycleLR":
        steps_per_epoch = len(dataloader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=1e-2,
            epochs=MAX_EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.25,)
    elif SCHEDULER == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-6,
            last_epoch=-1)
    elif SCHEDULER == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=2,
            gamma=0.1)
    else:
        scheduler = None

    return optimizer, scheduler


def get_device(n):
    if not PARALLEL_FOLD_TRAIN:
        n = 0
    if not USE_GPU and USE_TPU:
        return torch.device('cpu')
    elif USE_TPU:
        import torch_xla.core.xla_model as xm
        if not PARALLEL_FOLD_TRAIN:
            return xm.xla_device()
        else:
            return xm.xla_device(n)
    elif USE_GPU:
        return torch.device('cuda:' + str(n))
