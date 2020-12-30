#!/bin/bash
source activate pytorch_latest_p37
git init
git remote add origin https://KevinMathewT:skevin943@github.com/KevinMathewT/Cassava-Disease-Detection.git
git pull origin master
mkdir generated
/home/ubuntu/anaconda3/envs/pytorch_latest_p37/bin/python -m pip install --upgrade pip
pip install tqdm
pip install --upgrade opencv-python
pip install --upgrade timm
pip install adabelief_pytorch
pip install ranger_adabelief
pip install albumentations
lsblk
sudo file -s /dev/xvdf
sudo mkdir /vol
sudo mount /dev/xvdf /vol
sudo df -hT
sudo rsync -ah --progress /vol/. ./input/
python -m src.create_folds
git pull origin master
python -m src.train



source activate pytorch_latest_p37
git init
git remote add origin https://KevinMathewT:skevin943@github.com/KevinMathewT/Cassava-Disease-Detection.git
git pull origin master
mkdir generated
/home/ubuntu/anaconda3/envs/pytorch_latest_p37/bin/python -m pip install --upgrade pip
pip install --upgrade opencv-python
pip install --upgrade timm
pip install adabelief_pytorch
pip install ranger_adabelief
pip install albumentations
pip install kaggle
mkdir input
cd input
export KAGGLE_USERNAME="kevinmathewt"
export KAGGLE_KEY="4f1da67682fd7154f26f6691dbdfe0a3"
kaggle competitions download -c cassava-leaf-disease-classification
mkdir cassava-leaf-disease-classification
unzip cassava-leaf-disease-classification.zip -d cassava-leaf-disease-classification
cd ..
sudo df -hT
sudo rsync -ah --progress /vol/. ./input/
python -m src.create_folds
git pull origin master
python -m src.train
