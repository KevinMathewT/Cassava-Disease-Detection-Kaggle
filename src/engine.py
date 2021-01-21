import time
from src.utils import get_accuracy
from sklearn.metrics import accuracy_score
import numpy as np

import torch
from torch import optim
from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief

from . import config
from .models.models import *
from .utils import AccuracyMeter, AverageLossMeter

if config.USE_TPU:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp


def train_one_epoch(fold, epoch, model, loss_fn, optimizer, train_loader, device, scaler, scheduler=None, schd_batch_update=False):
    model.train()

    print_fn = print if not config.USE_TPU else xm.master_print
    t = time.time()
    running_loss = AverageLossMeter()
    running_accuracy = AccuracyMeter()
    total_steps = len(train_loader)
    pbar = enumerate(train_loader)
    optimizer.zero_grad()

    for step, (imgs, image_labels) in pbar:
        imgs, image_labels = imgs.to(device, dtype=torch.float32), image_labels.to(device, dtype=torch.int64)
        curr_batch_size = imgs.size(0)

        # print(image_labels.shape, exam_label.shape)
        if (not config.USE_TPU) and config.MIXED_PRECISION_TRAIN:
            with torch.cuda.amp.autocast():
                image_preds = model(imgs)
                loss = loss_fn(image_preds, image_labels)
                running_loss.update(
                    curr_batch_avg_loss=loss.item(), batch_size=curr_batch_size)
                running_accuracy.update(
                    y_pred=image_preds.detach().cpu(),
                    y_true=image_labels.detach().cpu() if not config.ONE_HOT_LABEL else torch.argmax(image_labels, 1).detach().cpu(),
                    batch_size=curr_batch_size)

            scaler.scale(loss).backward()

            if ((step + 1) % config.ACCUMULATE_ITERATION == 0) or ((step + 1) == total_steps):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step(epoch + (step / total_steps))

        else:
            image_preds = model(imgs)
            # print(xm.get_ordinal(), "Outputs:", torch.any(image_preds.isnan()))
            loss = loss_fn(image_preds, image_labels)
            # print(xm.get_ordinal(), "Loss:", torch.any(loss.isnan()))
            loss.backward()
        #     # print("Loss: ", loss.item())

            if ((step + 1) % config.ACCUMULATE_ITERATION == 0) or ((step + 1) == total_steps):
                if config.USE_TPU:
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()
                if scheduler is not None and schd_batch_update:
                    scheduler.step(epoch + (step / total_steps))
                optimizer.zero_grad()

            running_loss.update(
                curr_batch_avg_loss=loss.item(), batch_size=curr_batch_size)
            running_accuracy.update(
                y_pred=image_preds.detach().cpu(),
                y_true=image_labels.detach().cpu() if not config.ONE_HOT_LABEL else torch.argmax(image_labels, 1).detach().cpu(),
                batch_size=curr_batch_size)

            # print("Loss Update:", running_loss.avg)
            # print("Acc Update:", running_loss.avg)

        if config.USE_TPU:
            loss = xm.mesh_reduce(
                'train_loss_reduce', running_loss.avg, lambda x: sum(x) / len(x))
            acc = xm.mesh_reduce(
                'train_acc_reduce', running_accuracy.avg, lambda x: sum(x) / len(x))
        else:
            loss = running_loss.avg
            acc = running_accuracy.avg
        if ((config.LEARNING_VERBOSE and (step + 1) % config.VERBOSE_STEP == 0)) or ((step + 1) == total_steps) or ((step + 1) == 1):
            description = f'[{fold}/{config.FOLDS - 1}][{epoch:>2d}/{config.MAX_EPOCHS - 1:>2d}][{step + 1:>4d}/{total_steps:>4d}] Loss: {loss:.4f} | Accuracy: {acc:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f} | Time: {time.time() - t:.4f}'
            print_fn(description, flush=True)

        # break
    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(fold, epoch, model, loss_fn, valid_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    print_fn = print if not config.USE_TPU else xm.master_print
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
        image_targets_all += [image_labels.detach().cpu().numpy() if not config.ONE_HOT_LABEL else torch.argmax(image_labels, 1).detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        running_loss.update(curr_batch_avg_loss=loss.item(),
                            batch_size=image_labels.shape[0])
        sample_num += image_labels.shape[0]

        if config.USE_TPU:
            loss = xm.mesh_reduce(
                'valid_loss_reduce', running_loss.avg, lambda x: sum(x) / len(x))
        else:
            loss = running_loss.avg
        if ((config.LEARNING_VERBOSE and (step + 1) % config.VERBOSE_STEP == 0)) or ((step + 1) == len(valid_loader)) or ((step + 1) == 1):
            description = f'[{fold}/{config.FOLDS - 1}][{epoch:>2d}/{config.MAX_EPOCHS - 1:>2d}][{step + 1:>4d}/{total_steps:>4d}] Validation Loss: {loss:.4f}'
            print_fn(description)

        # break

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    if config.USE_TPU:
        acc = xm.mesh_reduce('valid_acc_reduce', accuracy_score(
            image_targets_all, image_preds_all), lambda x: sum(x) / len(x))
    else:
        acc = accuracy_score(image_targets_all, image_preds_all)
    print_fn(
        f'[{fold}/{config.FOLDS - 1}][{epoch:>2d}/{config.MAX_EPOCHS - 1:>2d}] Validation Multi-Class Accuracy = {acc:.4f}')

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

    if config.FREEZE_BATCH_NORM:
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    if config.USE_TPU:
        net = xmp.MpModelWrapper(net)

    return net


def get_device(n):
    print_fn = print if not config.USE_TPU else xm.master_print
    if not config.PARALLEL_FOLD_TRAIN:
        n = 0

    if not config.USE_GPU and not config.USE_TPU:
        print_fn(f"Device:                      CPU")
        return torch.device('cpu')
    elif config.USE_TPU:
        print_fn(f"Device:                      TPU")
        if not config.PARALLEL_FOLD_TRAIN:
            return xm.xla_device()
        else:
            return xm.xla_device(n)
    elif config.USE_GPU:
        print_fn(
            f"Device:                      GPU ({torch.cuda.get_device_name(0)})")
        return torch.device('cuda:' + str(n))
