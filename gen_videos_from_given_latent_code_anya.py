# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
from scipy.interpolate import CubicSpline
import torch
from tqdm import tqdm
import mrcfile

import legacy
import PIL.Image

from camera_utils import LookAtPoseSampler, get_render_pose
from torch_utils import misc
from training.triplane import TriPlaneGenerator

from ignite.engine import Engine
from ignite.metrics import PSNR, SSIM
from torchvision import transforms

# to keep ignite pytorch format
def get_output(metrics_engine, output):
    return output[0], output[1]

# ----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

# ----------------------------------------------------------------------------

def gen_interp_video(G, latent, mp4: str,  w_frames=60 * 4, grid_dims=(1, 1),
                     num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', focal=4.2647, image_mode='image',
                     gen_shapes=False, gen_imgs=True, target_img=None, device=torch.device('cuda'), noise_mode='const', **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    name = mp4[:-4]
    if num_keyframes is None:

        num_keyframes = 1 // (grid_w * grid_h)

    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)

    if cfg == 'FFHQ':
        cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
        focal = 4.2647
        print("Using geometry of dataset FFHQ for rendering.")
    elif cfg == 'Shapenet':
        cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
        focal = 1.025390625
        print("Using geometry of dataset Shapenet for rendering.")
    elif cfg == 'Carla':
        cam2world_pose = get_render_pose(radius=10.5, phi=0, theta=45).unsqueeze(0).to(device)
        focal = 1.8660254037844388
        phi = 0 # starting angle for rotation
        print("Using geometry of dataset Carla for rendering.")
    elif cfg == 'drr':
        cam2world_pose = get_render_pose(radius=10.5, phi=0, theta=45).unsqueeze(0).to(device)
        #focal = 1.5 # 1.5 for chest, 3.02 for knees
        phi = 0 # starting angle for rotation
        print("Using geometry of dataset DRR for rendering.")
    else:
        print("Geometry of dataset not recognised")

    intrinsics = torch.tensor([[focal, 0, 0.5], [0, focal, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(latent.shape[0], 1)
    ws = latent # 1, 14, 512

    if ws.shape[1] != G.backbone.mapping.num_ws:
        ws = ws.repeat([1,G.backbone.mapping.num_ws, 1])

    _ = G.synthesis(ws[:1], c[:1])  # warm up

    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Interpolation
    grid = []

    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1]) # (5, 14, 512) - might be different, depending on the model
            interpolator = CubicSpline(x, y, axis=0)
            interp = interpolator(np.linspace(-num_keyframes * wraps, num_keyframes * (wraps + 1), num=1))
            row.append(interp)
        grid.append(row)

    # Render video.
    max_batch = 10000000
    voxel_resolution = 512
    video_out = imageio.get_writer(mp4, mode='I', fps=30, codec='libx264', **video_kwargs)

    if gen_shapes:
        outdir = 'interpolation_{}/'.format(name)
        os.makedirs(outdir, exist_ok=True)
    all_poses = []
    ssim_value = 0.
    psnr_value = 0.
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                if cfg == 'FFHQ' or cfg == 'Shapenet':
                    pitch_range = 0.25
                    yaw_range = 0.35
                    cam2world_pose = LookAtPoseSampler.sample(
                    3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                    3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                    camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device) # torch.Size([1, 4, 4]), cuda
                elif cfg == 'drr' or cfg == 'Carla':
                    cam2world_pose = get_render_pose(radius=10.5, phi=phi + 360/(num_keyframes * w_frames) * frame_idx, theta=45).unsqueeze(0).to(device)
                
                all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                interp = grid[yi][xi] # grid[0][0].shape = (1, 14, 512); grid is a list
                w = torch.from_numpy(interp).to(device)

                if noise_mode == 'const':
                    img = G.synthesis(ws=w, c=c)[image_mode][0]
                else:
                    img = G.synthesis(ws=w, c=c)[image_mode][0]

                xray_recons = img

                if image_mode == 'image_depth':
                    img = -img
                    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1
                
                if frame_idx == 0: # just 1st image
                            psnr_engine = Engine(get_output)
                            psnr = PSNR(data_range=2.)
                            psnr.attach(psnr_engine, "psnr")
                            ssim_engine = Engine(get_output)
                            ssim = SSIM(data_range=2.)
                            ssim.attach(ssim_engine, "ssim")

                            data = torch.unsqueeze(torch.stack([xray_recons.unsqueeze(0), target_img.to(device)],0),0)
                            psnr_state = psnr_engine.run(data)
                            psnr_value += psnr_state.metrics['psnr']
                            ssim_state = ssim_engine.run(data)
                            ssim_value += ssim_state.metrics['ssim']
                            print(f"SSIM: {ssim_value:.4f} , PSNR: {psnr_value:.4f}")

                if gen_imgs == True:
                        PIL.Image.fromarray(layout_grid(img.unsqueeze(0)), 'RGB').save(mp4[:mp4.find('final/')+6]+'generated_'+'{:04d}.png'.format(frame_idx))

                imgs.append(img)

                if gen_shapes:
                    # generate shapes
                    print('Generating shape for frame %d / %d ...' % (frame_idx, num_keyframes * w_frames))

                    samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, voxel_origin=[0, 0, 0],
                                                                       cube_length=G.rendering_kwargs['box_warp'])
                    samples = samples.to(device)
                    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
                    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
                    transformed_ray_directions_expanded[..., -1] = -1

                    head = 0
                    with tqdm(total=samples.shape[1]) as pbar:
                        with torch.no_grad():
                            while head < samples.shape[1]:
                                torch.manual_seed(0)
                                sigma = G.sample_mixed(samples[:, head:head + max_batch],
                                                       transformed_ray_directions_expanded[:, :samples.shape[1] - head],
                                                       w.unsqueeze(0), truncation_psi=psi, noise_mode='const')['sigma']
                                sigmas[:, head:head + max_batch] = sigma
                                head += max_batch
                                pbar.update(max_batch)

                    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
                    sigmas = np.flip(sigmas, 0)

                    pad = int(30 * voxel_resolution / 256)
                    pad_top = int(38 * voxel_resolution / 256)
                    sigmas[:pad] = 0
                    sigmas[-pad:] = 0
                    sigmas[:, :pad] = 0
                    sigmas[:, -pad_top:] = 0
                    sigmas[:, :, :pad] = 0
                    sigmas[:, :, -pad:] = 0

                    output_ply = True
                    if output_ply:
                        from shape_utils import convert_sdf_samples_to_ply
                        convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1,
                                                   os.path.join(outdir, f'{frame_idx:04d}_shape.ply'), level=10)
                    else:  # output mrc
                        with mrcfile.new_mmap(outdir + f'{frame_idx:04d}_shape.mrc', overwrite=True, shape=sigmas.shape,
                                              mrc_mode=2) as mrc:
                            mrc.data[:] = sigmas
        
        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))

    video_out.close()
    all_poses = np.stack(all_poses)

    if gen_shapes:
        print(all_poses.shape)
        with open(mp4.replace('.mp4', '_trajectory.npy'), 'wb') as f:
            np.save(f, all_poses)


# ----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.
    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')


# ----------------------------------------------------------------------------

# @click.command()
# @click.option('--network', help='PTI Network pickle filename or original EG3D Network pickle filename', required=False)
# @click.option('--npy_path', 'npy_path', help='Network pickle filename', required=True)
# @click.option('--num-keyframes', type=int,
#               help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.',
#               default=None)
# @click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
# @click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
# @click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
# @click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
# @click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'Cats', 'Shapenet', 'Carla', 'drr']), required=False, metavar='STR',
#               default='FFHQ', show_default=True)
# @click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']),
#               required=False, metavar='STR', default='image', show_default=True)
# @click.option('--sample_mult', 'sampling_multiplier', type=float,
#               help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
# @click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
# @click.option('--foc', type=float, help='Focal length', default=4.2647, show_default=True)
# @click.option('--gen_imgs', help='Save interpolation images from video', required=False, is_flag=True, default=True, show_default=True)

def generate_images(
        network: str,
        npy_path:str,
        target: str,
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

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network)
    device = torch.device('cuda')

    if 'pkl' in network:
        print("Loading original EG3D Network")
        with dnnlib.util.open_url(network) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

        G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
        G.rendering_kwargs['depth_resolution_importance'] = int(
            G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    else:
        print("Loading finetuned PTI Network")
        G = torch.load(network)
        G.eval()

    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0  # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14  # no truncation so doesn't matter where we cutoff

    grid = (1,1)
    latent  = np.load(npy_path)
    latent = torch.FloatTensor(latent).cuda()
    transform_list = [transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    trans = transforms.Compose(transform_list)
    target_img = torch.unsqueeze(trans(PIL.Image.open(target).convert('RGB')), 0)

    name = os.path.basename(npy_path)[:-4]
    output = os.path.join(outdir, f'{name}.mp4')
    print(f'Generate {output}...')
    gen_interp_video(G=G, latent=latent, mp4=output, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes,
                     w_frames=w_frames,  psi=truncation_psi,
                     truncation_cutoff=truncation_cutoff, cfg=cfg, focal=foc, image_mode=image_mode, gen_imgs=gen_imgs, target_img=target_img, noise_mode=noise_mode)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------