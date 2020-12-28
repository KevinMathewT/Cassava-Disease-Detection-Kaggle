import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .config import *

if __name__ == "__main__":
    train = pd.read_csv(TRAIN)
    train["fold"] = -1
    skf = StratifiedKFold(n_splits=FOLDS)
    X = train['image_id']
    y = train["label"]

    for fold, (train_index, test_index) in enumerate(tqdm(skf.split(X, y))):
        train.loc[test_index, "fold"] = fold

    train.to_csv(TRAIN_FOLDS, index=False)

    print(f"{FOLDS} folds created and saved at: {TRAIN_FOLDS}.")    