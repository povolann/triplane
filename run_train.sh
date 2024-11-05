#!/bin/bash

# Set LD_PRELOAD before starting Python
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Run the Python training script
python train.py --outdir=./debug --cfg=drr --data=./datasets/chest_128.zip --gpus=1 --batch=8 --gamma=0.3 --z_dim=512 --lazy_reg=False
