#!/bin/bash
source activate pytorch_latest_p37
lsblk
sudo file -s /dev/xvdf
sudo mkdir /vol
sudo mount /dev/xvdf /vol
sudo df -hT
sudo cp -a /vol/. ./input/

