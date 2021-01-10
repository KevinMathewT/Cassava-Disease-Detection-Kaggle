import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from . import config

if __name__ == "__main__":
    train = pd.read_csv(config.TRAIN)
    train["fold"] = -1
    skf = StratifiedKFold(n_splits=config.FOLDS, shuffle=True, random_state=config.SEED)
    X = train['image_id']
    y = train["label"]

    for fold, (train_index, test_index) in tqdm(enumerate(skf.split(X, y))):
        train.loc[test_index, "fold"] = fold

    train.to_csv(config.TRAIN_FOLDS, index=False)

    print(f"{config.FOLDS} folds created and saved at: {config.TRAIN_FOLDS}.")