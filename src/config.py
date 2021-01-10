import os
import random
import numpy as np

import torch

# INPUT_PATH            = "./input"  # PC and EC2
INPUT_PATH            = "../input" # Kaggle
# INPUT_PATH            = "."        # Colab
GENERATED_FILES_PATH  = "./generated/"
DATASET_PATH          = os.path.join(INPUT_PATH, "cassava-leaf-disease-classification/")
TRAIN                 = os.path.join(DATASET_PATH, "train.csv")
TRAIN_IMAGES_DIR      = os.path.join(DATASET_PATH, "train_images")
TEST_IMAGES_DIR       = os.path.join(DATASET_PATH, "test_images")
TEST                  = os.path.join(DATASET_PATH, "sample_submission.csv")
TRAIN_FOLDS           = os.path.join(GENERATED_FILES_PATH, "train_folds.csv")
WEIGHTS_PATH          = "./generated/weights/" # For PC and Kaggle
# WEIGHTS_PATH          = "/content/drive/My Drive" # For Colab
# WEIGHTS_PATH          = "/vol/weights/" # For EC2
USE_GPU               = True
USE_TPU               = False
GPUS                  = 1
TPUS                  = 8 # Basically TPU Nodes
PARALLEL_FOLD_TRAIN   = False
SEED                  = 719
FOLDS                 = 5
MIXED_PRECISION_TRAIN = False

DO_DEPTH_MASKING      = False
DO_FMIX               = False
DO_CUTMIX             = False

H                     = 512
W                     = 512

OPTIMIZER             = "Adam" # [Adam, AdamW, AdaBelief, RangerAdaBelief]
SCHEDULER             = "CosineAnnealingWarmRestarts" # [ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts, StepLR]
TRAIN_CRITERION       = "BiTemperedLogisticLoss" # [BiTemperedLogisticLoss, SoftmaxCrossEntropy, FocalCosineLoss, SmoothCrossEntropyLoss]
VALID_CRITERION       = "SoftmaxCrossEntropy" # [SoftmaxCrossEntropy, FocalCosineLoss, SmoothCrossEntropyLoss]
LEARNING_RATE         = 1e-4
MAX_EPOCHS            = 15

N_CLASSES             = 5

TRAIN_BATCH_SIZE      = 32
VALID_BATCH_SIZE      = 32
ACCUMULATE_ITERATION  = 2

NET                   = "seresnext50_32x4d" # [SEResNeXt50_32x4d_BH, ResNeXt50_32x4d_BH, ViTBase16_BH, 
                                             #  resnext50_32x4d, seresnext50_32x4d, tf_efficientnet_b4_ns, gluon_resnet18_v1b]

PRETRAINED            = True
LEARNING_VERBOSE      = True
VERBOSE_STEP          = 100

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


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)