import time
from joblib import Parallel, delayed

import torch
import torch.nn as nn

from .dataset import get_loaders
from .engine import get_device, get_net, get_optimizer_and_scheduler, train_one_epoch, valid_one_epoch
from .config import *
from .utils import *
from .loss import FocalCosineLoss, SmoothCrossEntropyLoss, bi_tempered_logistic_loss

if USE_TPU:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl

import warnings
warnings.filterwarnings("ignore")


def run_fold(fold):
    create_dirs()
    print_fn = print if not USE_TPU else xm.master_print
    print_fn(f"Training Fold:               {fold}")
    print_fn(f"Mixed Precision Training:    {MIXED_PRECISION_TRAIN}")

    global net
    net                             = xmp.MpModelWrapper(net) if USE_TPU else net
    train_loader, valid_loader      = get_loaders(fold)
    device                          = get_device(n=fold+1)
    mp_device_loader                = pl.MpDeviceLoader(train_loader, 
                                                        device, fixed_batch_size=True) if USE_TPU else None
    net                             = net.to(device)
    scaler                          = torch.cuda.amp.GradScaler() if not USE_TPU and MIXED_PRECISION_TRAIN else None
    # loss_tr                         = nn.CrossEntropyLoss().to(device)  # MyCrossEntropyLoss().to(device)
    # loss_tr                         = FocalCosineLoss(device=device).to(device)
    # loss_tr                         = SmoothCrossEntropyLoss(smoothing=0.1).to(device)
    loss_tr                         = bi_tempered_logistic_loss
    loss_fn                         = nn.CrossEntropyLoss().to(device)
    optimizer, scheduler            = get_optimizer_and_scheduler(net=net, 
                                                                  dataloader=train_loader)

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        train_one_epoch(fold, epoch, net, loss_tr, optimizer, train_loader, device, scaler=scaler,
                        scheduler=scheduler, schd_batch_update=False)
        with torch.no_grad():
            valid_one_epoch(fold, epoch, net, loss_fn, valid_loader,
                            device, scheduler=None, schd_loss_update=False)
        print_fn(f"Time Taken for Epoch {epoch}: {time.time() - epoch_start}")

        if USE_TPU:
            xm.save(net.state_dict(
            ), os.path.join(WEIGHTS_PATH, f'{NET}/{NET}_fold_{fold}_{epoch}.bin'))
        else:
            torch.save(net.state_dict(
            ), os.path.join(WEIGHTS_PATH, f'{NET}/{NET}_fold_{fold}_{epoch}.bin'))

    #torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
    del net, optimizer, train_loader, valid_loader, scheduler
    torch.cuda.empty_cache()


def _mp_fn(rank, flags):
    torch.set_default_tensor_type("torch.FloatTensor")
    for fold in [1, 3]:
        global net
        net = get_net(name=NET, pretrained=PRETRAINED)
        a = run_fold(fold)


def train():
    print_fn = print
    print_fn(f"Training Model:              {NET}")

    if not USE_TPU:
        if not PARALLEL_FOLD_TRAIN:
            # for fold in range(2, FOLDS):
            #     run_fold(fold)
            # run_fold(0)
            # for fold in [0]:
            #     global net
            #     net = get_net(name=NET, pretrained=PRETRAINED)
            #     run_fold(fold)

            config.NET = "ViTBase16_BH"
            config.H   = 224
            config.W   = 224
            config.ACCUMULATE_ITERATION = 1
            config.TRAIN_BATCH_SIZE = 32
            config.VALID_BATCH_SIZE = 32

            for fold in [0]:
                global net
                net = get_net(name=NET, pretrained=PRETRAINED)
                run_fold(fold)

        if PARALLEL_FOLD_TRAIN:
            n_jobs = FOLDS
            parallel = Parallel(n_jobs=n_jobs, backend="threading")
            parallel(delayed(run_fold)(fold) for fold in range(FOLDS))

    if USE_TPU:
        if MIXED_PRECISION_TRAIN:
            os.environ["XLA_USE_BF16"] = "1"
        os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"

        FLAGS = {}
        xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method="fork")


if __name__ == "__main__":
    train()
