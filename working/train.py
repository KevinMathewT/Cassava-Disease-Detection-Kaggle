import torch
import torch.nn as nn

from .dataset import get_loaders
from .engine import get_net, get_optimizer_and_scheduler, train_one_epoch, valid_one_epoch
from .config import *
from .utils import *

import warnings
warnings.filterwarnings("ignore")


def run_fold(fold):
    train_loader, valid_loader = get_loaders(fold)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_net(name=NET, pretrained=PRETRAINED).to(device)
    scaler = torch.cuda.amp.GradScaler()
    optimizer, scheduler = get_optimizer_and_scheduler(net=net)

    loss_tr = nn.CrossEntropyLoss().to(device)  # MyCrossEntropyLoss().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    create_dirs()

    for epoch in range(MAX_EPOCHS):
        train_one_epoch(epoch, net, loss_tr, optimizer, train_loader, device, scaler=scaler,
                        scheduler=scheduler, schd_batch_update=False)

        with torch.no_grad():
            valid_one_epoch(epoch, net, loss_fn, valid_loader,
                            device, scheduler=None, schd_loss_update=True)

        torch.save(net.state_dict(
        ), f'./working/models/weights/{NET}/{NET}_fold_{fold}_{epoch}')

    #torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
    del net, optimizer, train_loader, valid_loader, scheduler
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"Training Model : {NET}")

    for fold in range(FOLDS):
        run_fold(fold)

    # if PARALLEL_FOLD_TRAIN:
        # n_jobs = FOLDS
        # predictions = np.concatenate(Parallel(n_jobs=n_jobs, backend="threading")(
        #     delayed(run_fold)(fold) for fold in range(FOLDS)), axis=0)

    # else:
    # predictions = np.concatenate([run_fold(fold) for fold in range(FOLDS)], axis=0)

    # predictions = pd.DataFrame(predictions, columns=[
    #                         "image_id", "0", "1", "2", "3", "4"])
    # predictions.to_csv(os.path.join(GENERATED_FILES_PATH,
    #                                 f"{NET}-predictions.csv"), index=False)
