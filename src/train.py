from joblib import Parallel, delayed

import torch
import torch.nn as nn

from .dataset import get_loaders
from .engine import get_device, get_net, get_optimizer_and_scheduler, train_one_epoch, valid_one_epoch
from .config import *
from .utils import *

import warnings
warnings.filterwarnings("ignore")


def run_fold(fold):
    print(f"Training Fold: {fold}")

    train_loader, valid_loader = get_loaders(fold)
    device = get_device(n=fold+1)
    net = get_net(name=NET, pretrained=PRETRAINED).to(device)
    scaler = torch.cuda.amp.GradScaler()
    optimizer, scheduler = get_optimizer_and_scheduler(net=net, dataloader=train_loader)

    loss_tr = nn.CrossEntropyLoss().to(device)  # MyCrossEntropyLoss().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(MAX_EPOCHS):
        train_one_epoch(fold, epoch, net, loss_tr, optimizer, train_loader, device, scaler=scaler,
                        scheduler=scheduler, schd_batch_update=True)
        with torch.no_grad():
            valid_one_epoch(fold, epoch, net, loss_fn, valid_loader,
                            device, scheduler=None, schd_loss_update=True)

        if USE_TPU:
             xm.save(net.state_dict(
            ), os.path.join(WEIGHTS_PATH, f'{NET}/{NET}_fold_{fold}_{epoch}.bin'))
        else:
            torch.save(net.state_dict(
            ), os.path.join(WEIGHTS_PATH, f'{NET}/{NET}_fold_{fold}_{epoch}.bin'))

    #torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
    del net, optimizer, train_loader, valid_loader, scheduler
    torch.cuda.empty_cache()


def train():
    print(f"Training Model : {NET}")

    if not PARALLEL_FOLD_TRAIN:
        # for fold in range(FOLDS):
        #     run_fold(fold)
        run_fold(3)

    if PARALLEL_FOLD_TRAIN:
        n_jobs = FOLDS
        parallel = Parallel(n_jobs=n_jobs, backend="threading")
        parallel(delayed(run_fold)(fold) for fold in range(FOLDS))


if __name__ == "__main__":
    create_dirs()
    train()
