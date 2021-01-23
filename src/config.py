import os
import random
import numpy as np

import torch

# INPUT_PATH            = "./input"  # PC and EC2
# INPUT_PATH            = "../input" # Kaggle
INPUT_PATH            = "."        # Colab
GENERATED_FILES_PATH  = "./generated/"
DATASET_PATH          = os.path.join(INPUT_PATH, "cassava-leaf-disease-classification/")
TRAIN_IMAGES_DIR      = os.path.join(DATASET_PATH, "train_images")
TEST_IMAGES_DIR       = os.path.join(DATASET_PATH, "test_images")
TRAIN                 = os.path.join(DATASET_PATH, "train.csv")
TEST                  = os.path.join(DATASET_PATH, "sample_submission.csv")
TRAIN_FOLDS           = os.path.join(GENERATED_FILES_PATH, "train_folds.csv")
# WEIGHTS_PATH          = "./generated/weights/" # For PC and Kaggle
WEIGHTS_PATH          = "/content/drive/My Drive/weights" # For Colab
# WEIGHTS_PATH          = "/vol/weights/" # For EC2
USE_GPU               = True # [True, False]
USE_TPU               = False # [True, False]
GPUS                  = 1
TPUS                  = 8 # Basically TPU Nodes
PARALLEL_FOLD_TRAIN   = False # [True, False]
SEED                  = 719
FOLDS                 = 5
MIXED_PRECISION_TRAIN = True # [True, False]
DROP_LAST             = True # [True, False]
DO_FREEZE_BATCH_NORM  = True # [True, False]
FREEZE_BN_EPOCHS      = 5


ONE_HOT_LABEL         = False
DO_DEPTH_MASKING      = False
DO_FMIX               = False
FMIX_PROBABILITY      = 0.5
DO_CUTMIX             = False
CUTMIX_PROBABILITY    = 0.5


H                     = 512 # [224, 384, 512]
W                     = 512 # [224, 384, 512]

OPTIMIZER             = "RAdam"  # [Adam, AdamW, RAdam, AdaBelief, RangerAdaBelief]
SCHEDULER             = "CosineAnnealingWarmRestarts" # [ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts, StepLR]
SCHEDULER_WARMUP      = True # [True, False]
WARMUP_EPOCHS         = 1 if SCHEDULER_WARMUP else 0
WARMUP_FACTOR         = 7 if SCHEDULER_WARMUP else 1
TRAIN_CRITERION       = "BiTemperedLogisticLoss" # [BiTemperedLogisticLoss, LabelSmoothingCrossEntropy, SoftmaxCrossEntropy, FocalCosineLoss, SmoothCrossEntropyLoss, TaylorCrossEntropyLoss, RandomChoice]
VALID_CRITERION       = "SoftmaxCrossEntropy" # [BiTemperedLogisticLoss, SoftmaxCrossEntropy, FocalCosineLoss, SmoothCrossEntropyLoss, TaylorCrossEntropyLoss, RandomChoice]
LEARNING_RATE         = 1e-4
MAX_EPOCHS            = 15
SCHEDULER_BATCH_STEP  = True # [True, False]

N_CLASSES             = 5

TRAIN_BATCH_SIZE      = 32
VALID_BATCH_SIZE      = 16
ACCUMULATE_ITERATION  = 2

NET                   = "EfficientNetB4" # [SEResNeXt50_32x4d_BH, ResNeXt50_32x4d_BH, ViTBase16_BH, ViTBase16, ViTLarge16
                                             #  resnext50_32x4d, seresnext50_32x4d, tf_efficientnet_b4_ns, ['vit_base_patch16_224', 'vit_base_patch16_384', 'vit_base_patch32_384', 'vit_base_resnet26d_224', 'vit_base_resnet50d_224', 'vit_huge_patch16_224', 'vit_huge_patch32_384', 'vit_large_patch16_224', 'vit_large_patch16_384', 'vit_large_patch32_384', 'vit_small_patch16_224', 'vit_small_resnet26d_224', 'vit_small_resnet50d_s3_224']

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


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)