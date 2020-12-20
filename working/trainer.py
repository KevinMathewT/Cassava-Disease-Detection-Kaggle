import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch

from .config import *


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []
        self.lr = None

    def on_validation_end(self, trainer, pl_module):
        for scheduler in trainer.lr_schedulers:
            param_groups = scheduler['scheduler'].optimizer.param_groups
            lr = param_groups[0]["lr"]
            if self.lr is None:
                print(f"Start Learning Rate is {lr}")
            elif lr < self.lr:
                print(f"Learning Rate Reduced to {lr}")
            else:
                print(f"Learning Rate Increased to {lr}")
            self.lr = lr
        self.metrics.append(trainer.callback_metrics)


def get_trainer(net, fold, name):
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="tb_logs",
        name=f"base_fold_{fold}",
        version=0
    )

    lr_monitor = LearningRateMonitor(
        logging_interval='step'
    )

    metrics_callback = MetricsCallback()

    checkpoint_callback = ModelCheckpoint(
        filepath=f'./working/models/{name}/cldc-net={name}-fold={fold}-' +
        '{epoch:03d}-{val_loss_epoch:.4f}',
        monitor='val_loss_epoch',
        save_top_k=SAVE_TOP_K,
        verbose=False
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        mode='min',
        patience=EARLY_STOPPING,
        verbose=False
    )

    callbacks = [lr_monitor, metrics_callback]

    if USE_EARLY_STOPPING:
        callbacks.append(early_stop_callback)

    tpu_cores = [fold + 1] if PARALLEL_FOLD_TRAIN else 8

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        logger=tb_logger,
        max_epochs=MAX_EPOCHS,
        gpus=torch.cuda.device_count() if torch.cuda.is_available() else None,
        # tpu_cores=tpu_cores,
        precision=16,
        callbacks=callbacks,
        # progress_bar_refresh_rate=1,
        checkpoint_callback=checkpoint_callback,  # Do not save any checkpoints,
    )
    trainer.use_native_amp = False

    return trainer, checkpoint_callback, metrics_callback
