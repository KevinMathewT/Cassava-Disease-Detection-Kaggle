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
    import torch_xla.distributed.xla_multiprocessing as xmp


def train_one_epoch(fold, epoch, model, loss_fn, optimizer, train_loader, device, scaler, scheduler=None, schd_batch_update=False):
    model.train()

    print_fn = print if not USE_TPU else xm.master_print
    t = time.time()
    running_loss = AverageLossMeter()
    running_accuracy = AccuracyMeter()
    total_steps = len(train_loader)
    pbar = enumerate(train_loader)
    optimizer.zero_grad()

    for step, (imgs, image_labels) in pbar:
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
            # print(xm.get_ordinal(), "Outputs:", torch.any(image_preds.isnan()))
            loss = loss_fn(image_preds, image_labels)
            # print(xm.get_ordinal(), "Loss:", torch.any(loss.isnan()))
            loss.backward()
        #     # print("Loss: ", loss.item())

            if ((step + 1) % ACCUMULATE_ITERATION == 0) or ((step + 1) == total_steps):
                if USE_TPU:
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()
                if scheduler is not None and schd_batch_update:
                    scheduler.step()
                optimizer.zero_grad()

            running_loss.update(
                curr_batch_avg_loss=loss.item(), batch_size=curr_batch_size)
            running_accuracy.update(
                y_pred=image_preds.detach().cpu(),
                y_true=image_labels.detach().cpu(),
                batch_size=curr_batch_size)

            # print("Loss Update:", running_loss.avg)
            # print("Acc Update:", running_loss.avg)

        if USE_TPU:
            loss = xm.mesh_reduce(
                'train_loss_reduce', running_loss.avg, lambda x: sum(x) / len(x))
            acc = xm.mesh_reduce(
                'train_acc_reduce', running_accuracy.avg, lambda x: sum(x) / len(x))
        else:
            loss = running_loss.avg
            acc = running_accuracy.avg
        if ((LEARNING_VERBOSE and (step + 1) % VERBOSE_STEP == 0)) or ((step + 1) == total_steps) or ((step + 1) == 1):
            description = f'[{fold}/{FOLDS - 1}][{epoch:>2d}/{MAX_EPOCHS - 1}][{step + 1:>4d}/{total_steps:>4d}] Loss: {loss:.4f} | Accuracy: {acc:.4f} | Time: {time.time() - t:.4f}'
            print_fn(description, flush=True)

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

        if USE_TPU:
            loss = xm.mesh_reduce(
                'valid_loss_reduce', running_loss.avg, lambda x: sum(x) / len(x))
        else:
            loss = running_loss.avg
        if ((LEARNING_VERBOSE and (step + 1) % VERBOSE_STEP == 0)) or ((step + 1) == len(valid_loader)) or ((step + 1) == 1):
            description = f'[{fold}/{FOLDS - 1}][{epoch:>2d}/{MAX_EPOCHS - 1}][{step + 1:>4d}/{total_steps:>4d}] Validation Loss: {loss:.4f}'
            print_fn(description)

        # break

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    if USE_TPU:
        acc = xm.mesh_reduce('valid_acc_reduce', accuracy_score(
            image_targets_all, image_preds_all), lambda x: sum(x) / len(x))
    else:
        acc = accuracy_score(image_targets_all, image_preds_all)
    print_fn(
        f'[{fold}/{FOLDS - 1}][{epoch}/{MAX_EPOCHS - 1}] Validation Multi-Class Accuracy = {acc:.4f}')

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

    return net


def get_optimizer_and_scheduler(net, dataloader):
    print_fn = print if not USE_TPU else xm.master_print
    # m = xm.xrt_world_size() if USE_TPU else 1
    m = 1
    print_fn(f"World Size:                  {m}")

    # Optimizers

    print_fn(f"Optimizer:                   {OPTIMIZER}")
    if OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(
            params=net.parameters(),
            lr=LEARNING_RATE * m,
            weight_decay=1e-5,
            amsgrad=False
        )
    elif OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(
            net.parameters(), lr=LEARNING_RATE * m, weight_decay=0.001)
    elif OPTIMIZER == "AdaBelief":
        optimizer = AdaBelief(net.parameters(
        ), lr=LEARNING_RATE * m, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False, print_change_log=False)
    elif OPTIMIZER == "RangerAdaBelief":
        optimizer = RangerAdaBelief(
            net.parameters(), lr=LEARNING_RATE * m, eps=1e-12, betas=(0.9, 0.999), print_change_log=False)
    else:
        optimizer = optim.SGD(
            net.parameters(), lr=LEARNING_RATE * m)

    # Schedulers

    print_fn(f"Scheduler:                   {SCHEDULER}")
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
    print_fn = print if not USE_TPU else xm.master_print
    if not PARALLEL_FOLD_TRAIN:
        n = 0

    if not USE_GPU and not USE_TPU:
        print_fn(f"Device:                      CPU")
        return torch.device('cpu')
    elif USE_TPU:
        print_fn(f"Device:                      TPU")
        if not PARALLEL_FOLD_TRAIN:
            return xm.xla_device()
        else:
            return xm.xla_device(n)
    elif USE_GPU:
        print_fn(f"Device:                      GPU ({torch.cuda.get_device_name(0)})")
        return torch.device('cuda:' + str(n))
