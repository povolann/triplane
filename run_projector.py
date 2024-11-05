
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
import numpy as np
import torch
import legacy
from torchvision.transforms import transforms
from projector import w_projector_anya, w_plus_projector_anya
from PIL import Image
# ----------------------------------------------------------------------------


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
# @click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# @click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
# @click.option('--latent_space_type', help='latent_space_type', type=click.Choice(['w', 'w_plus']), required=False, metavar='STR',
#               default='w', show_default=True)
# @click.option('--image_path', help='image_path', type=str, required=True, metavar='STR', show_default=True)
# @click.option('--c_path', help='camera parameters path', type=str, required=True, metavar='STR', show_default=True)
# @click.option('--sample_mult', 'sampling_multiplier', type=float,
#               help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
# @click.option('--num_steps', 'num_steps', type=int,
#               help='Number of steps for training', default=2000, show_default=True)
# @click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
# @click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'Cats', 'Shapenet', 'Carla', 'drr']), required=False, metavar='STR',
#               default='FFHQ', show_default=True)
# @click.option('--foc', type=float, help='Focal length', default=4.2647, show_default=True)

def run(
        network_pkl: str,
        outdir: str,
        sampling_multiplier: float,
        nrr: Optional[int],
        latent_space_type:str,
        image_path:str,
        c_path:str,
        num_steps:int,
        cfg: str,
        foc: float,
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

    os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    if network_pkl.endswith('.pkl'):  # use legacy network pkl
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    elif network_pkl.endswith('.pth'):  # use pth
        G = torch.load(network_pkl)['G_ema']
    else:
        raise ValueError(f'Invalid network pickle: {network_pkl}')

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr


    image = Image.open(image_path).convert('RGB')
    image_name = os.path.basename(image_path)[:-4]
    c = np.load(c_path) # c.shape = (25,)
    c = np.reshape(c,(1,25))

    c = torch.FloatTensor(c).cuda()

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Resize((G.img_resolution,G.img_resolution)) #512, 512 changed to G.img_resolution
    ])
    from_im = trans(image).cuda()
    id_image = torch.squeeze((from_im.cuda() + 1) / 2) * 255 # W & H must match G output resolution

    if latent_space_type == 'w':
        w = w_projector_anya.project(G, c, outdir, id_image, device=torch.device('cuda'), w_avg_samples=600, num_steps = num_steps, w_name=image_name, cfg=cfg, focal=foc, noise_mode=noise_mode)
    else:
        w = w_plus_projector_anya.project(G, c, outdir, id_image, device=torch.device('cuda'), w_avg_samples=600, w_name=image_name, num_steps = num_steps, cfg=cfg, focal=foc)
        pass

    w = w.detach().cpu().numpy()
    np.save(f'{outdir}/{image_name}_{latent_space_type}/{image_name}_{latent_space_type}.npy', w)

    PTI_embedding_dir = f'./projector/PTI/embeddings/{image_name}'
    os.makedirs(PTI_embedding_dir,exist_ok=True)
    np.save(f'./projector/PTI/embeddings/{image_name}/{image_name}_{latent_space_type}.npy', w)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------



