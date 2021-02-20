# Cassava Disease Detection

My pipeline for the [Cassava Disease Detection](https://www.kaggle.com/c/cassava-leaf-disease-classification/) competition on Kaggle.

This solution was top 10% (Bronze Medal).

### Leaderboard Scores:
| Model                            | Public | Private |
| -------------------------------- |:------:| -------:|
| SEResNeXt50 + Binary Head        | 0.900  | 0.897   |
| ResNeXt50 + EfficientNextB3      | 0.900  | 0.896   |

My best solution was a single model SEResNeXt50 with a Binary Head.

## Training
To train the model set the configurations in `src/config.py`.
1. To train on GPUs set `USE_GPU = True` and `USE_TPU = False`, and vice versa for training on TPUs.
   * To train on CPUs set `USE_GPU = False` and `USE_TPU = False`
   * In my limited experience, training on TPUs gave worse CV and LB scores compared to training on GPUs, all my submissions were GPU trained (This was a problem, since I only had Kaggle and Google Colab resources to train my models.)
2. To use Automatic Mixed Precision set `MIXED_PRECISION_TRAIN = True`, reduces the training time, results were more or less the same.
3. To use FMix, or CutMix, set `ONE_HOT_LABEL = True`, and `DO_FMIX = True` or `DO_CUTMIX = True`, and also set the probabilities.
4. Freezing Batch Normalization layers in the first few epochs improved my scores. To do this, set `DO_FREEZE_BATCH_NORM = True` and `FREEZE_BN_EPOCHS` to the number of starting epochs to keep frozen.
5. Warming up the Learning Rate linearly also improved the score. To do this set `SCHEDULER_WARMUP = True`, and `WARMUP_EPOCHS` and `WARMUP_FACTOR` by with appropriate values.

The folds have already been generated in `./generated/train_folds.csv`.

You can also change the number of folds and generate it by running:
```
python -m src.create_folds
```
After the configurations are set, and the folds are ready, train the model by running:
```
python -m src.train
```


## Training Configurations for SEResNeXt50:
```python

SEED                  = 719
FOLDS                 = 5
MIXED_PRECISION_TRAIN = True
DROP_LAST             = True
DO_FREEZE_BATCH_NORM  = True
FREEZE_BN_EPOCHS      = 5

TRAIN_BATCH_SIZE      = 32
VALID_BATCH_SIZE      = 16
ACCUMULATE_ITERATION  = 2

ONE_HOT_LABEL         = False
DO_DEPTH_MASKING      = False
DO_FMIX               = False
DO_CUTMIX             = False

H                     = 512
W                     = 512

OPTIMIZER             = "RAdam"
SCHEDULER             = "CosineAnnealingWarmRestarts"
SCHEDULER_WARMUP      = True
WARMUP_EPOCHS         = 1
WARMUP_FACTOR         = 7
TRAIN_CRITERION       = "BiTemperedLogisticLoss"
VALID_CRITERION       = "SoftmaxCrossEntropy"
LEARNING_RATE         = 1e-4
MAX_EPOCHS            = 15
SCHEDULER_BATCH_STEP  = True

NET                   = "SEResNeXt50_32x4d_BH"
```

### Augmentations
The Albumentations library was used for all Augmentations.
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
