#!/bin/bash
source activate pytorch_latest_p37
lsblk
sudo file -s /dev/xvdf
sudo mkdir /vol
sudo mount /dev/xvdf /vol
sudo df -hT
sudo rsync -ah --progress /vol/. ./input/
git init
git remote add origin https://KevinMathewT:skevin943@github.com/KevinMathewT/Cassava-Disease-Detection.git
git pull origin master
mkdir generated
