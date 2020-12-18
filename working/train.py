from .dataset import get_train_dataloader, get_valid_dataloader
from .engine import get_net
from .trainer import get_trainer
from .config import *
from .utils import *
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.special import softmax
import warnings
warnings.filterwarnings("ignore")


def run_fold(fold, predictions):
    train = train_folds[train_folds.fold != fold]
    valid = train_folds[train_folds.fold == fold]

    train_loader = get_train_dataloader(
        train.drop(['fold'], axis=1).to_numpy())
    valid_loader = get_valid_dataloader(
        valid.drop(['fold'], axis=1).to_numpy())

    net = get_net(name=NET, fold=fold, pretrained=PRETRAINED)
    trainer, checkpoint_callback, metrics_callback = get_trainer(
        net=net, fold=fold, name=NET)

    trainer.fit(net, train_loader, valid_loader)

    print(
        f"Best Model for Fold #{fold} saved at {checkpoint_callback.best_model_path}.")

    net = get_net(name=NET, fold=fold, pretrained=PRETRAINED)
    net.load_state_dict(torch.load(
        checkpoint_callback.best_model_path)["state_dict"])
    net.to("cuda")
    pred_ids = valid.image_id.to_numpy().reshape(-1, 1)
    pred_ids = pred_ids[:SUBSET_SIZE, :] if USE_SUBSET else pred_ids
    print("Prediction IDs: ", len(pred_ids))

    preds = []
    with torch.no_grad():
        for data in tqdm(valid_loader):
            image, _ = data
            image = image.cuda()
            outputs = softmax(net(image).cpu().detach().numpy(), axis=1)
            preds.append(outputs)

    preds = np.concatenate(preds, axis=0)
    preds = np.concatenate([pred_ids, preds], axis=1)
    predictions = np.concatenate([predictions, preds], axis=0)

    return predictions


if __name__ == "__main__":
    train_folds = pd.read_csv(TRAIN_FOLDS)

    print(f"Training Model : {NET}")
    predictions = np.empty((0, 6))

    for fold in range(1):
        predictions = run_fold(fold, predictions)

    predictions = pd.DataFrame(predictions, columns=[
                               "image_id", "0", "1", "2", "3", "4"])
    predictions.to_csv(os.path.join(GENERATED_FILES_PATH,
                                    f"{NET}-predictions.csv"), index=False)
