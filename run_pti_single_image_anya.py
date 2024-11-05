from random import choice
from string import ascii_uppercase
from torchvision.transforms import transforms
import os
#from configs import global_config, paths_config_anya # original
from projector.PTI.configs import global_config, paths_config_anya # run_all
import glob
import click

from projector.PTI.training.coaches.single_image_coach import SingleImageCoach # run_all

# @click.command()

# @click.option('--run_name', help='Run name', required=False, metavar='STR', default='', show_default=True)
# @click.option('--use_wandb', help='Use wandb', required=False, is_flag=True, default=False, show_default=True)
# @click.option('--use_multi_id_training', help='Use multi id training', required=False, is_flag=True, default=False, show_default=True)

# @click.option('--network_name', help='Network pickle filename', required=True)
# @click.option('--image_in', help='Image input folder name', required=False, metavar='STR', default='FFHQ', show_default=True)
# @click.option('--latent_space_type', help='latent_space_type', type=click.Choice(['w', 'w_plus']), required=False, metavar='STR',
#               default='w', show_default=True)
#@click.option('--outdir_proj', help='Latent code filename', required=False, metavar='STR')

def run_PTI(run_name, use_wandb, use_multi_id_training, network_name, image_in, latent_space_type, outdir_proj, noise_mode):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name


    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    global_config.network_name = network_name


    # All paths are saved in paths_config.py file
    #embedding_dir_path = f'{paths_config_anya.embedding_base_dir}/{image_in}/{paths_config_anya.pti_results_keyword}'
    #os.makedirs(embedding_dir_path, exist_ok=True)

    image_dir = f'{paths_config_anya.input_data_path}{image_in}'

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    coach = SingleImageCoach(trans)

    if latent_space_type == 'w_plus':
        for image_path in glob.glob(f'{image_dir}/*.png'):
            name = os.path.basename(image_path)[:-4]
            #w_path = f'{paths_config_anya.base_dir}projector_out/{name}_{latent_space_type}/{name}_{latent_space_type}.npy'
            if outdir_proj is None:
                w_path = f'{paths_config_anya.base_dir}projector_out/{name}_{latent_space_type}/{name}_{latent_space_type}.npy'
            else:
                w_path = os.path.join(outdir_proj, f'{name}_{latent_space_type}/{name}_{latent_space_type}.npy')
            c_path = image_path.replace('png', 'npy')
            #if len(glob.glob(f'{paths_config_anya.checkpoints_dir}/*_{name}_{latent_space_type}.pth')) > 0:
            #    continue

            if not os.path.exists(w_path):
                continue
            coach.train(image_path=image_path, w_path=w_path,c_path=c_path)

    if latent_space_type == 'w':
        for image_path in glob.glob(f'{image_dir}/*.png'):
            name = os.path.basename(image_path)[:-4]
            #w_path = f'{paths_config_anya.base_dir}projector_out/{name}_{latent_space_type}/{name}_{latent_space_type}.npy'
            if outdir_proj is None:
                w_path = f'{paths_config_anya.base_dir}projector_out/{name}_{latent_space_type}/{name}_{latent_space_type}.npy'
            else:
                w_path = os.path.join(outdir_proj, f'{name}_{latent_space_type}/{name}_{latent_space_type}.npy')            
            c_path = image_path.replace('png', 'npy')
            #if len(glob.glob(f'{paths_config_anya.checkpoints_dir}/*_{name}_{latent_space_type}.pth')) > 0:
            #    continue

            if not os.path.exists(w_path):
                continue
            coach.train(image_path=image_path, w_path=w_path, c_path=c_path, noise_mode=noise_mode)

    return global_config.run_name


if __name__ == '__main__':
    run_PTI()
