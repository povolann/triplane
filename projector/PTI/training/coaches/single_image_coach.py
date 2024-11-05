import os
import torch
from tqdm import tqdm
# from configs import paths_config_anya, hyperparameters, global_config # original
# from training.coaches.base_coach import BaseCoach
# from utils.log_utils import log_images_from_w

from projector.PTI.configs import paths_config_anya, hyperparameters, global_config # run_all
from projector.PTI.training.coaches.base_coach import BaseCoach
from projector.PTI.utils.log_utils import log_images_from_w
import numpy as np
from PIL import Image

class SingleImageCoach(BaseCoach):

    def __init__(self,trans):
        super().__init__(data_loader=None, use_wandb=False)
        self.source_transform = trans

    def train(self, image_path, w_path, c_path, noise_mode, G_helper=None):

        use_ball_holder = True

        name = os.path.basename(w_path)[:-4]
        print("image_path: ", image_path, 'c_path', c_path)
        c = np.load(c_path)

        c = np.reshape(c, (1, 25))

        c = torch.FloatTensor(c).cuda()

        from_im = Image.open(image_path).convert('RGB')

        if self.source_transform:
            image = self.source_transform(from_im)

        self.restart_training()




        print('load pre-computed w from ', w_path)
        if not os.path.isfile(w_path):
            print(w_path, 'is not exist!')
            return None

        w_pivot = torch.from_numpy(np.load(w_path)).to(global_config.device)


        # w_pivot = w_pivot.detach().clone().to(global_config.device)
        w_pivot = w_pivot.to(global_config.device)

        log_images_counter = 0
        #real_images_batch = image.to(global_config.device)
        real_images_batch = image.to(global_config.device).unsqueeze(0)

        for i in tqdm(range(hyperparameters.max_pti_steps)):

            generated_images = self.forward(w_pivot, c, noise_mode)
            loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, name,
                                                           self.G, use_ball_holder, w_pivot)

            self.optimizer.zero_grad()

            if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                break

            loss.backward()
            self.optimizer.step()

            use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0


            global_config.training_step += 1
            log_images_counter += 1

        self.image_counter += 1

        #save_dict = {'G_ema': self.G.state_dict()}
        checkpoint_path = f'{paths_config_anya.checkpoints_dir}/model_{global_config.run_name}_{name}.pth'
        print('final model ckpt save to ', checkpoint_path)
        torch.save(self.G, checkpoint_path)

        # import pickle
        # import copy
        # import dnnlib
        # import training.legacy
        # with dnnlib.util.open_url(paths_config_anya.eg3d_ffhq_pkl.replace('name', global_config.network_name)) as f:
        #      G_helper = training.legacy.load_network_pkl(f)['G_ema'].cpu()  # type: ignore #.to(device)

        # module = copy.deepcopy(self.G).eval().requires_grad_(False).cpu()
        # G_helper['G_ema'] = module
        
        # with open(checkpoint_path.replace('pth', 'pkl'), 'wb') as f:
        #             pickle.dump(G_helper, f)