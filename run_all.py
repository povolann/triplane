"""Generate videos using pretrained network pickle from 1 given image."""

import os
from typing import List, Optional, Tuple, Union

import click
import glob
from convert_pkl_2_pth_anya import convert
from run_projector import run
from run_pti_single_image_anya import run_PTI
from gen_videos_from_given_latent_code_anya import generate_images
from projector.PTI.configs import paths_config_anya
import time

@click.command()
@click.option('--network_pkl', help='Network pickle filename', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--num-keyframes', type=int,
              help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.',
              default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=72)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory for final video and images', type=str, required=True, metavar='DIR')
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'Cats', 'Shapenet', 'Carla', 'drr']), required=False, metavar='STR',
              default='drr', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']),
              required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--foc', type=float, help='Focal length', default=4.2647, show_default=True)
@click.option('--latent_space_type', help='latent_space_type', type=click.Choice(['w', 'w_plus']), required=False, metavar='STR',
              default='w', show_default=True)
@click.option('--num_steps', 'num_steps', type=int, help='Number of steps for for w or w+ projector training', default=200, show_default=True)
@click.option('--run_name', help='Run name', required=False, metavar='STR', default='', show_default=True)
@click.option('--use_wandb', help='Use wandb', required=False, is_flag=True, default=False, show_default=True)
@click.option('--image_in', help='Image input folder name', required=False, metavar='STR', default='chest', show_default=True)
@click.option('--gen_imgs', help='Save interpolation images from video', required=False, is_flag=True, default=True, show_default=True)
@click.option('--noise_mode', help='Noise in Synthesis network', type=click.Choice(['random', 'const', 'none']), required=False, metavar='STR', default='const', show_default=True)

def generate(
        network_pkl: str,
        shuffle_seed: Optional[int],
        truncation_psi: float,
        truncation_cutoff: int,
        num_keyframes: Optional[int],
        w_frames: int,
        outdir: str,
        cfg: str,
        image_mode: str,
        sampling_multiplier: float,
        nrr: Optional[int],
        foc: float,
        latent_space_type: str,
        num_steps:int,
        run_name:str,
        use_wandb:bool,
        image_in:str,
        gen_imgs:bool,
        noise_mode: str,
):
    """Render a latent vector interpolation video.
    Examples:
    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
    Animation length and seed keyframes:
    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.
    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.
    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """
    start_time = time.time()

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    image_dir = f'./projector_test_data/{image_in}'
    outdir_proj = os.path.join(outdir, 'projector_out')

    # print('Starting with conversion of the model to pth')
    # outdir_conv = os.path.join(outdir, 'convert_pkl_2_pth_out')
    # network_pth = convert(network_pkl=network_pkl, shuffle_seed=shuffle_seed, truncation_psi=truncation_psi,
    #         truncation_cutoff=truncation_cutoff, num_keyframes=num_keyframes, w_frames=w_frames, outdir=outdir_conv, cfg=cfg,
    #         image_mode=image_mode, sampling_multiplier=sampling_multiplier, nrr=nrr, foc=foc)
    # print('Model converted to pth and saved at:', network_pth)

    print(f'Starting with projector training for latent space type: {latent_space_type} with {num_steps} steps')
    for image_path in glob.glob(f'{image_dir}/*.png'):
        c_path = image_path.replace('png', 'npy')
        run(network_pkl=network_pkl, outdir=outdir_proj, sampling_multiplier=sampling_multiplier, nrr=nrr, latent_space_type=latent_space_type,
        image_path=image_path, c_path=c_path, num_steps=num_steps, cfg=cfg, foc=foc, noise_mode=noise_mode)
        print('Projector training completed for', image_path)

    # PTI runs for the all given images in image_in folder already
    print('Starting with PTI training')
    network_name = network_pkl.replace('.pkl', '').replace('./networks/', '')
    run_name = run_PTI(run_name='', use_wandb=use_wandb, use_multi_id_training=False, network_name=network_name, image_in=image_in, latent_space_type=latent_space_type,
                       outdir_proj=outdir_proj, noise_mode=noise_mode)
    print('PTI training completed and models saved with name:', run_name)

    outdir_fin = os.path.join(outdir, 'final')
    for image_path in glob.glob(f'{image_dir}/*.png'):
        name = os.path.basename(image_path)[:-4]
        w_path = f'{outdir_proj}/{name}_{latent_space_type}/{name}_{latent_space_type}.npy'
        target = f'{outdir_proj}/{name}_{latent_space_type}/{num_steps-100}.png'
        if run_name == '':
            run_name = 'FSECGHZMZDEY'
        network_pti = os.path.join(paths_config_anya.checkpoints_dir, f'model_{run_name}_{name}_{latent_space_type}.pth')
        generate_images(network=network_pti, npy_path=w_path, target=target, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
            num_keyframes=num_keyframes, w_frames=w_frames, outdir=outdir_fin, cfg=cfg, image_mode=image_mode, sampling_multiplier=sampling_multiplier,
            nrr=nrr, foc=foc, gen_imgs=gen_imgs, noise_mode=noise_mode)
        print('Video (+ images) generated and saved at:', outdir)
        tick_end_time = time.time()
        print(f'Time taken for generating video for {name}: {tick_end_time - start_time} seconds')

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate()

# ----------------------------------------------------------------------------