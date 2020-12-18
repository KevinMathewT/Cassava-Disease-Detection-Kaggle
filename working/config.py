import os
from pytorch_lightning import seed_everything

INPUT_PATH           = "../input/"
GENERATED_FILES_PATH = "./generated/"
DATASET_PATH         = os.path.join(INPUT_PATH, "cassava-leaf-disease-classification/")
TRAIN                = os.path.join(DATASET_PATH, "train.csv")
TRAIN_IMAGES_DIR     = os.path.join(DATASET_PATH, "train_images")
TEST_IMAGES_DIR      = os.path.join(DATASET_PATH, "test_images")
TEST                 = os.path.join(DATASET_PATH, "sample_submission.csv")
TRAIN_FOLDS          = os.path.join(GENERATED_FILES_PATH, "train_folds.csv")
GPUS                 = 1

H                    = 512
W                    = 512

LEARNING_RATE        = 1e-4
WEIGHT_DECAY         = 1e-5
MAX_EPOCHS           = 30

N_CLASSES            = 5
TRAIN_BATCH_SIZE     = 16
VALID_BATCH_SIZE     = 16
USE_SUBSET           = False
SUBSET_SIZE          = TRAIN_BATCH_SIZE * 1
CPU_WORKERS          = 4
NET                  = 'SEResNeXt50_32x4d'
# NET                  = 'gluon_resnet18_v1b'

SEED                 = 2020
FOLDS                = 8
SEEDS                = 1

PRETRAINED           = True
SAVE_TOP_K           = 1
LEARNING_VERBOSE     = False
EARLY_STOPPING       = 10
USE_EARLY_STOPPING   = False

seed_everything(SEED)