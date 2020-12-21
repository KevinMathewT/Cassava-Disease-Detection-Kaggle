import os
from .utils import seed_everything

# INPUT_PATH           = "./input"  # PC
# INPUT_PATH           = "../input" # Kaggle
INPUT_PATH           = "."        # Colab
GENERATED_FILES_PATH = "./generated/"
DATASET_PATH         = os.path.join(INPUT_PATH, "cassava-leaf-disease-classification/")
TRAIN                = os.path.join(DATASET_PATH, "train.csv")
TRAIN_IMAGES_DIR     = os.path.join(DATASET_PATH, "train_images")
TEST_IMAGES_DIR      = os.path.join(DATASET_PATH, "test_images")
TEST                 = os.path.join(DATASET_PATH, "sample_submission.csv")
TRAIN_FOLDS          = os.path.join(GENERATED_FILES_PATH, "train_folds.csv")
# WEIGHTS_PATH         = "./working/models/weights"
WEIGHTS_PATH         = "/content/drive/My Drive"
USE_GPU              = False
USE_TPU              = True
GPUS                 = 1
TPUS                 = 8
PARALLEL_FOLD_TRAIN  = True
SEED                 = 719
FOLDS                = 5
SEEDS                = 1
MIXED_PRECISION_TRAIN= True

H                    = 512
W                    = 512

OPTIMIZER            = "AdaBelief"
SCHEDULER            = "CosineAnnealingLR"
LEARNING_RATE        = 1e-4
MAX_EPOCHS           = 10

N_CLASSES            = 5

ACCUMULATE_ITERATION = 16
TRAIN_BATCH_SIZE     = 2
VALID_BATCH_SIZE     = 32

NET                  = 'SEResNeXt50_32x4d'
# NET                  = 'resnext50_32x4d'
# NET                  = 'tf_efficientnet_b4_ns'
# NET                  = 'gluon_resnet18_v1b'

PRETRAINED           = True
SAVE_TOP_K           = 1
LEARNING_VERBOSE     = True
VERBOSE_STEP         = 1
EARLY_STOPPING       = 10
USE_EARLY_STOPPING   = False


USE_SUBSET           = False
SUBSET_SIZE          = TRAIN_BATCH_SIZE * 1
CPU_WORKERS          = 0

# if not PARALLEL_FOLD_TRAIN:
#     TRAIN_BATCH_SIZE //= TPUS
#     VALID_BATCH_SIZE //= TPUS
seed_everything(SEED)