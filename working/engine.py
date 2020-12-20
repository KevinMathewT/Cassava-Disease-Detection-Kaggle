# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.modules.loss import _WeightedLoss
# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim

# import pytorch_lightning as pl
# from pytorch_lightning import seed_everything
# from pytorch_lightning import loggers as pl_loggers
# from pytorch_lightning import Callback
# from pytorch_lightning.metrics import Accuracy
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from .config import *
# from .utils import *
# from .loss import get_train_criterion, get_valid_criterion
# from .models.models import nets, GeneralizedCassavaClassifier

# from adabelief_pytorch import AdaBelief
# from ranger_adabelief import RangerAdaBelief


# class CassavaLitModule(pl.LightningModule):
#     def __init__(self, net, fold):
#         super(CassavaLitModule, self).__init__()

#         self.net = net
#         self.fold = fold

#         self.train_criterion = get_train_criterion()
#         self.valid_criterion = get_valid_criterion()
#         self.train_losses = []
#         self.valid_losses = []
#         self.epoch = 0
#         self.best_valid_loss = None
#         self.current_epoch_train_loss = None
#         self.accuracy = pl.metrics.Accuracy()

#     def forward(self, x):
#         return self.net(x)

#     def configure_optimizers(self):
#         if OPTIMIZER == "Adam":
#             optimizer = torch.optim.Adam(
#                 params=self.parameters(),
#                 lr=LEARNING_RATE,
#                 weight_decay=1e-5
#             )
#         elif OPTIMIZER == "AdamW":
#             optimizer = optim.AdamW(
#                 self.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
#         elif OPTIMIZER == "AdaBelief":
#             optimizer = AdaBelief(self.parameters(
#             ), lr=LEARNING_RATE, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False)
#         elif OPTIMIZER == "RangerAdaBelief":
#             optimizer = RangerAdaBelief(
#                 self.parameters(), lr=LEARNING_RATE, eps=1e-12, betas=(0.9, 0.999))
#         else:
#             optimizer = optim.SGD(
#                 self.parameters(), lr=LEARNING_RATE)

#         if SCHEDULER == "ReduceLROnPlateau":
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer,
#                 patience=0,
#                 factor=0.1,
#                 verbose=LEARNING_VERBOSE
#             )
#             return {
#                 'optimizer': optimizer,
#                 'lr_scheduler': scheduler,
#                 'monitor': 'val_loss'
#             }

#         elif SCHEDULER == "CosineAnnealingLR":
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#                 optimizer, T_max=5, eta_min=0)

#             return [optimizer], [scheduler]

#         elif SCHEDULER == "OneCycleLR":
#             steps_per_epoch = len(self.train_dataloader())
#             # steps_per_epoch = len(self.train_dataloader())//self.trainer.accumulate_grad_batches
#             scheduler = torch.optim.lr_scheduler.OneCycleLR(
#                 optimizer=optimizer,
#                 pct_start=0.1,
#                 div_factor=1e3,
#                 max_lr=1e-2,
#                 steps_per_epoch=steps_per_epoch,
#                 epochs=MAX_EPOCHS
#             )
#             scheduler = {"scheduler": scheduler, "interval": "step"}
#             return [optimizer], [scheduler]
#         else:
#             return optimizer

#     def training_step(self, batch, batch_idx):
#         inputs, targets = batch
#         outputs = self(inputs)

#         loss = self.train_criterion(outputs, targets)
#         accuracy = self.accuracy(outputs, targets)

#         self.log('train_loss', loss, on_step=True,
#                  on_epoch=True, prog_bar=True, logger=True)
#         self.log("train_acc", accuracy, on_step=True,
#                  on_epoch=True, prog_bar=True, logger=True)
#         self.train_losses.append(loss.item())
#         return loss

#     def training_epoch_end(self, outputs):
#         losses = 0.0
#         for output in outputs:
#             loss = output["loss"]
#             losses += loss.item()

#         losses /= len(outputs)

#         self.current_epoch_train_loss = losses

#     def validation_step(self, batch, batch_idx):
#         inputs, targets = batch
#         outputs = self(inputs)

#         loss = self.valid_criterion(outputs, targets)
#         accuracy = self.accuracy(outputs, targets)

#         self.log('val_loss', loss, on_step=True,
#                  on_epoch=True, prog_bar=True, logger=True)
#         self.log("val_acc", accuracy, on_step=True,
#                  on_epoch=True, prog_bar=True, logger=True)
#         self.valid_losses.append(loss.item())
#         return loss

#     def validation_epoch_end(self, outputs):
#         losses = 0.0
#         for output in outputs:
#             loss = output
#             losses += loss.item()

#         losses /= len(outputs)

#         self.log('val_loss_epoch', losses, on_epoch=True,
#                  prog_bar=False, logger=True)

#         if self.best_valid_loss is not None:
#             self.best_valid_loss = min(self.best_valid_loss, losses)
#         else:
#             self.best_valid_loss = losses

#         if self.current_epoch_train_loss is None:
#             print(f"[{self.fold+1:>1}/{FOLDS:>1}][Epoch: {self.epoch:>2}] Training Loss: Sanity Check | Validation Loss: {losses:>.10f}")
#         else:
#             print(f"[{self.fold+1:>1}/{FOLDS:>1}][Epoch: {self.epoch:>2}] Training Loss: {self.current_epoch_train_loss:>.10f} | Validation Loss: {losses:>.10f}")
#         self.epoch += 1

#     def teardown(self, stage):
#         train_fold_loss = sum(self.train_losses) / len(self.train_losses)
#         valid_fold_loss = sum(self.valid_losses) / len(self.valid_losses)
#         best_valid_fold_loss = self.best_valid_loss
#         print(f"[{self.fold+1:>1}/{FOLDS:>1}] Training Loss: {train_fold_loss:>.10f} | Validation Loss: {valid_fold_loss:>.10f} | Best Validation Loss: {best_valid_fold_loss:>.10f}")

import time
from working.utils import get_accuracy
from tqdm import tqdm
import numpy as np

import torch
from torch import optim
from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief

from .dataset import CassavaDataset
from .config import *
from .models.models import *


def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scaler, scheduler=None, schd_batch_update=False):
    model.train()

    t = time.time()
    running_loss = None
    total_steps = len(train_loader)

    pbar = tqdm(enumerate(train_loader), total=total_steps)
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        #print(image_labels.shape, exam_label.shape)
        with torch.cuda.amp.autocast():
            image_preds = model(imgs)
            loss = loss_fn(image_preds, image_labels)
            accuracy = get_accuracy(image_preds, image_labels)

        scaler.scale(loss).backward()
        running_loss = loss.item() if running_loss is None else (
            running_loss * .99 + loss.item() * .01)

        if ((step + 1) % ACCUMULATE_ITERATION == 0) or ((step + 1) == total_steps):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None and schd_batch_update:
                scheduler.step()

        if ((LEARNING_VERBOSE and (step + 1) % VERBOSE_STEP == 0)) or ((step + 1) == total_steps):
            description = f'[{epoch}/{MAX_EPOCHS}][{step+1}/{total_steps}] Loss: {running_loss:.4f} | Accuracy: {accuracy:.4f}'
            pbar.set_description(description)

    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(epoch, model, loss_fn, valid_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

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
            description = f'[{epoch}/{MAX_EPOCHS}] | Validation Loss:{loss_sum/sample_num:.4f}'
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('[{epoch}/{MAX_EPOCHS}] Validation Multi-Class Accuracy = {:.4f}'.format(
        (image_preds_all == image_targets_all).mean()))

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
        ), lr=LEARNING_RATE, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False)
    elif OPTIMIZER == "RangerAdaBelief":
        optimizer = RangerAdaBelief(
            net.parameters(), lr=LEARNING_RATE, eps=1e-12, betas=(0.9, 0.999))
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
