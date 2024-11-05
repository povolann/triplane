# Triplane

## Related papers
- [Mednerf](https://arxiv.org/pdf/2202.01020 "Mednerf") and code: [https://github.com/abrilcf/mednerf] 
- [EG3D](https://arxiv.org/pdf/2112.07945 "EG3D") and code: [https://github.com/NVlabs/eg3d/tree/7cf1fd1e99e1061e8b6ba850f91c94fe56e7afe4]
- [ProjectedGAN](https://arxiv.org/pdf/2111.01007 "ProjectedGAN") and code: [https://github.com/autonomousvision/projected-gan]

  # Training
  To run the training with chest dataset use
  
  `python train.py --outdir=./training-runs --cfg=drr --data=./datasets/chest_128.zip --gpus=1 --batch=8 --gamma=0.3 --z_dim=256 --lazy_reg=True`

  # Inversion
  After training you can run the `run_all.py` script for generating new 72 images conditioned on 1 image
  
  `python run_all.py --network_pkl ./networks/network-snapshot-000000.pkl --trunc 0.7 --outdir training-runs/00000-drr-chest_128-gpus1-batch8-projected/inversion --cfg drr --foc 1.5 --image_in chest_psnr --num_steps 3000`

  Subsequently you call immediately calculate PSNR for these newly generated images.
  `python metrics.py --dataset chest --img_dir training-runs/00000-drr-chest_128-gpus1-batch8-projected/inversion`
