#!/bin/bash

# ./projector.sh

# Set LD_PRELOAD before starting Python
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Chest
python run_all.py --network_pkl ./networks/07.pkl --trunc 0.7 --outdir debug/00007-drr-chest_128-gpus1-batch8-projected/inversion --cfg drr --foc 1.5 --image_in chest_psnr --num_steps 3000
python metrics.py --dataset chest --img_dir debug/00007-drr-chest_128-gpus1-batch8-projected/inversion

#python run_all.py --network_pkl ./networks/correct_poses.pkl --trunc 0.7 --outdir correct_poses --cfg drr --foc 1.5 --image_in chest_psnr --num_steps 1000
#python metrics.py --dataset chest --img_dir correct_poses

# Knee
#python run_all.py --network_pkl ./networks/knee_2k_best.pkl --trunc 0.7 --outdir 2k/StyleGAN_best/knee_4500 --cfg drr --foc 3.02 --image_in knee_psnr --num_steps 4500

python run_all.py --network_pkl ./networks/carla/06.pkl --trunc 0.7 --outdir debug/00006-carla-carla_128-gpus1-batch8-projected/inversion --cfg Carla --foc 1.8660254037844388 --image_in carla --num_steps 3000



