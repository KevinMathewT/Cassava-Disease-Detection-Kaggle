import os
from .utils import seed_everything

# INPUT_PATH            = "./input"  # PC and EC2
# INPUT_PATH            = "../input" # Kaggle
INPUT_PATH            = "."        # Colab
GENERATED_FILES_PATH  = "./generated/"
DATASET_PATH          = os.path.join(INPUT_PATH, "cassava-leaf-disease-classification/")
TRAIN                 = os.path.join(DATASET_PATH, "train.csv")
TRAIN_IMAGES_DIR      = os.path.join(DATASET_PATH, "train_images")
TEST_IMAGES_DIR       = os.path.join(DATASET_PATH, "test_images")
TEST                  = os.path.join(DATASET_PATH, "sample_submission.csv")
TRAIN_FOLDS           = os.path.join(GENERATED_FILES_PATH, "train_folds.csv")
# WEIGHTS_PATH          = "./generated/weights/" # For PC and Kaggle
WEIGHTS_PATH          = "/content/drive/My Drive" # For Colab
# WEIGHTS_PATH          = "/vol/weights/" # For EC2
USE_GPU               = True
USE_TPU               = False
GPUS                  = 1
TPUS                  = 8 # Basically TPU Nodes
PARALLEL_FOLD_TRAIN   = False
SEED                  = 719
FOLDS                 = 5
SEEDS                 = 1
MIXED_PRECISION_TRAIN = True

H                     = 512
W                     = 512

OPTIMIZER             = "Adam"
SCHEDULER             = "CosineAnnealingWarmRestarts"
LEARNING_RATE         = 1e-4
MAX_EPOCHS            = 15

N_CLASSES             = 5

TRAIN_BATCH_SIZE      = 32
VALID_BATCH_SIZE      = 32
ACCUMULATE_ITERATION  = 1

NET                   = 'SEResNeXt50_32x4d+Binary_Head'
# NET                   = 'resnext50_32x4d'
# NET                   = 'tf_efficientnet_b4_ns'
# NET                   = 'gluon_resnet18_v1b'

PRETRAINED            = True
LEARNING_VERBOSE      = True
VERBOSE_STEP          = 1

USE_SUBSET            = False
SUBSET_SIZE           = TRAIN_BATCH_SIZE * 1
CPU_WORKERS           = 4

TRAIN_BATCH_SIZE    //= ACCUMULATE_ITERATION
VALID_BATCH_SIZE    //= ACCUMULATE_ITERATION
if not PARALLEL_FOLD_TRAIN:
    if USE_TPU:
        TRAIN_BATCH_SIZE //= TPUS
        VALID_BATCH_SIZE //= TPUS
    elif USE_GPU:
        TRAIN_BATCH_SIZE //= GPUS
        VALID_BATCH_SIZE //= GPUS

seed_everything(SEED)