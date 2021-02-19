# Cassava-Disease-Detection

My pipeline for the [Cassava Disease Detection](https://www.kaggle.com/c/cassava-leaf-disease-classification/leaderboard) competition on Kaggle.

This solution was top 11%.

### Leaderboard Scores:
| Model                            | Public | Private |
| -------------------------------- |:------:| -------:|
| SEResNeXt50                      | 0.900  | 0.897   |
| ResNeXt50 + EfficientNextB3      | 0.900  | 0.897   |

My best solution was a single model ResNext50.

## Training Configurations for ResNext50:
```python

SEED                  = 719
FOLDS                 = 5
MIXED_PRECISION_TRAIN = True
DROP_LAST             = True
DO_FREEZE_BATCH_NORM  = True
FREEZE_BN_EPOCHS      = 5

TRAIN_BATCH_SIZE      = 32
VALID_BATCH_SIZE      = 8
ACCUMULATE_ITERATION  = 2

ONE_HOT_LABEL         = False
DO_DEPTH_MASKING      = False
DO_FMIX               = False
DO_CUTMIX             = False

H                     = 512
W                     = 512

OPTIMIZER             = "RAdam"
SCHEDULER             = "CosineAnnealingWarmRestarts"
SCHEDULER_WARMUP      = True # [True, False]
WARMUP_EPOCHS         = 1
WARMUP_FACTOR         = 7
TRAIN_CRITERION       = "BiTemperedLogisticLoss"
VALID_CRITERION       = "SoftmaxCrossEntropy"
LEARNING_RATE         = 1e-4
MAX_EPOCHS            = 15
SCHEDULER_BATCH_STEP  = True # [True, False]
```

### Augmentations
#### Training
```python
Compose([
    RandomResizedCrop(config.H, config.W),
    Transpose(p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    ShiftScaleRotate(p=0.5),
    HueSaturationValue(hue_shift_limit=0.2,
                       sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    RandomBrightnessContrast(
        brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    Normalize(mean=[0.485, 0.456, 0.406], std=[
              0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    CoarseDropout(p=0.5),
    Cutout(p=0.5),
    ToTensorV2(p=1.0),
], p=1.)
```

#### Validation
```python
Compose([
    Resize(config.H, config.W),
    Normalize(mean=[0.485, 0.456, 0.406], std=[
              0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0),
], p=1.)
```


