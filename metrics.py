from ignite.engine import Engine
from ignite.metrics import PSNR, SSIM
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import dnnlib
import argparse

# to keep ignite pytorch format
def get_output(metrics_engine, output):
    return output[0], output[1]

def calculate_metrics(dir_1, dir_2):
    psnr_engine = Engine(get_output)
    psnr = PSNR(data_range=2.)
    psnr.attach(psnr_engine, "psnr")
    ssim_engine = Engine(get_output)
    ssim = SSIM(data_range=2.)
    ssim.attach(ssim_engine, "ssim")

    transform_list = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]
    trans = transforms.Compose(transform_list)

    total_psnr = 0.
    total_ssim = 0.

    for img in sorted(os.listdir(dir_1)):
        img_name = img[7:11]
        img_1 = torch.unsqueeze(trans(Image.open(os.path.join(dir_1, img)).convert('RGB')), 0)
        img_2 = torch.unsqueeze(trans(Image.open(os.path.join(dir_2, f'generated_{img_name}.png')).convert('RGB')), 0)
        psnr_state = psnr_engine.run([[img_1, img_2]])
        ssim_state = ssim_engine.run([[img_1, img_2]])
        print(psnr_state.metrics['psnr'], ssim_state.metrics['ssim'])
        print('PSNR for target image is', psnr_state.metrics['psnr'], 'and SSIM', ssim_state.metrics['ssim']) if img_name == '0000' else None

        total_psnr += psnr_state.metrics['psnr']
        total_ssim += ssim_state.metrics['ssim']

    num_images = len(os.listdir(dir_1))
    mean_psnr_value = total_psnr/num_images
    mean_ssim_value = total_ssim/num_images
    print('Mean PSNR is', mean_psnr_value, 'and mean SSIM is', mean_ssim_value, f'for {num_images} images')

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description='Calculate PSNR and SSIM between the ground truth and generated images.'
    )
    parser.add_argument('--dataset', type=str, default='knee', help='Dataset name')
    parser.add_argument('--img_dir', type=str, default='/home/anya/Programs/mednerf/graf-main/results/knee_pre/renderings_pre_over/', help='Name to dir where are generated images')

    args, unknown = parser.parse_known_args()

    dataset = 'knee'
    dir_1 = f'/home/anya/Programs/mednerf/graf-main/data/render/{args.dataset}' # ground truth data
    dir_2 = args.img_dir # MedNeRF model

    dnnlib.util.Logger(file_name=os.path.join(dir_2, 'metrics.txt'), file_mode='a', should_flush=True) # save everything to log.txt

    for file in os.scandir(dir_2):
        if file.is_dir():
            if file.name == 'final':
                print('Calculating metrics for', file.name, '...')
                calculate_metrics(dir_1, file)