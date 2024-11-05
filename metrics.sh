#!/bin/bash

# ./metrics.sh

#python metrics.py --dataset knee --img_dir /home/anya/Programs/triplane/2k/StyleGAN_1040/knee_4500/
#python metrics.py --dataset knee --img_dir /home/anya/Programs/triplane/2k/StyleGAN_1040/knee_5000/

python metrics.py --dataset chest --img_dir /home/anya/Programs/triplane/debug/best_model_triplane_new
python metrics.py --dataset chest --img_dir /home/anya/Programs/triplane/debug/best_model_triplane_new2
