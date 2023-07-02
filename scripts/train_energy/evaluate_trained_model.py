import os
import time

import torch
from torch.utils.data import DataLoader
from affordance_nets.datasets.pusht_diffusion_policy_data import PushTImageProjectedDataset
from affordance_nets.utils.directory_utils import makedirs
import torch.optim as optim
from tqdm import tqdm


import matplotlib.pyplot as plt


# GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(model):

    from affordance_nets.utils.directory_utils import get_data_dir
    path = get_data_dir()
    zarr_path = os.path.join(path, 'diffusion_policy', 'pusht', 'pusht_cchi_v7_replay.zarr')

    train_data = PushTImageProjectedDataset(zarr_path=zarr_path, horizon=5, resize=model.vision_model.img_size)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)


    with torch.no_grad():
        # Evaluation loop
        for i, batch in enumerate(train_dataloader):

            images = batch['obs']['image'][:, 0, ...]
            _x = batch['obs']['agent_pos'][:, 0, :].to(device)
            images = model.vision_model.image_preprocess(images).to(device)

            context = {'images': images}
            model.set_context(context)

            ## Generate energy for a 2D Grid ##
            x = torch.linspace(0, 1, images.shape[-1])
            yy, xx = torch.meshgrid((x,x))
            xy = torch.cat((xx.reshape(-1)[:,None], yy.reshape(-1)[:,None]), dim=-1).to(device)

            ## Given the massive size split in pieces and compute the energy
            energy = torch.zeros(xy.shape[0]).to(device)

            _n = 100
            batches = xy.shape[0]//_n
            context_features = model.context_features.repeat(_n, 1, 1)
            for k in range(batches):
                xy_p = xy[_n*k:_n*(k+1),:]

                model.context_features = context_features[:xy_p.shape[0],...]
                v = model(x=xy_p)
                dist = (v).pow(2).sum(-1).pow(.5)
                energy_p = torch.exp(-dist.pow(2) / 0.01)

                energy[_n*k:_n*(k+1)] = energy_p.squeeze()

            energy_grid = energy.reshape(xx.shape[0], xx.shape[1]).cpu().numpy()
            img_vis = images[0,...].permute(1,2,0).cpu().numpy()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())

            img_vis_e = .5*img_vis + .5*energy_grid[:,:,None]

            ## Visualize the Visual energy Maps ##
            _, ax = plt.subplots(1, 3, figsize=(8, 4))

            ax[0].imshow(img_vis)
            ax[1].imshow(energy_grid)
            ax[2].imshow(img_vis_e)

            plt.show()



if __name__ == '__main__':
    ## Save Torch Model ##
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'trained_models', 'pusht')
    checkpoint_model = os.path.join(checkpoint_dir, 'model.pth')

    from affordance_nets.models.main_models.implicit_film_energy_models import ImplicitFiLMImageEBM as Model
    from affordance_nets.models.vision_backbone.clip import CLIP_Backbone
    from affordance_nets.models.vision_backbone.resnet import ResNet18_Backbone

    vision_backbone = ResNet18_Backbone()
    model = Model(vision_backbone=vision_backbone).to(device)
    model.load_state_dict(torch.load(checkpoint_model))
    print('Loaded Model')

    ## Test the models segemntation maps ##
    main(model)