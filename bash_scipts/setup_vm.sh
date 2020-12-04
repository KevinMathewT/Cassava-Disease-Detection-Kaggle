#!/bin/bash
source activate pytorch_latest_p37
lsblk
file -s /dev/xvdf
mount /dev/xvdf /vol
df -hT
mv /vol/ ./input/