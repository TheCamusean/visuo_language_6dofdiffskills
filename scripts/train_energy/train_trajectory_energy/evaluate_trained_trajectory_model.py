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


def main(model,horizon):

    from affordance_nets.utils.directory_utils import get_data_dir
    path = get_data_dir()
    zarr_path = os.path.join(path, 'diffusion_policy', 'pusht', 'pusht_cchi_v7_replay.zarr')

    train_data = PushTImageProjectedDataset(zarr_path=zarr_path, horizon=horizon, resize=model.vision_model.img_size)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)


    with torch.no_grad():
        # Evaluation loop
        for i, batch in enumerate(train_dataloader):

            images = batch['obs']['image'][:, 0, ...]
            _x = batch['obs']['agent_pos'][:, :, :].to(device)
            images = model.vision_model.image_preprocess(images).to(device)

            context = {'images': images}
            model.set_context(context)

            ## Sample ##
            start_time = time.time()
            x = model.sample(B=2, T=100, alpha_0=1e-1)
            print('Compute time: {}'.format(time.time() - start_time))

            _x = (x*images.shape[-1]).cpu().numpy()

            img = images[0,...].permute(1,2,0).cpu().numpy()
            img_vis = (img - img.min()) / (img.max() - img.min())

            for k in range(_x.shape[0]):
                img2 = img_vis.copy()
                trj = _x[k,...]
                img2[int(trj[0,1]), int(trj[0,0]), :] = [0, 0, 0]

                H = 10
                for l in range(1, H):
                    agent_pos_pred = trj[l, :]
                    img2[int(agent_pos_pred[1]), int(agent_pos_pred[0])] = [0, 1, 0]

                ## Visualize the Visual energy Maps ##
                _, ax = plt.subplots(1, 2, figsize=(8, 4))

                ax[0].imshow(img_vis)
                ax[1].imshow(img2)

                plt.show()



if __name__ == '__main__':

    horizon = 10
    ## Save Torch Model ##
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'trained_models', 'pusht')
    checkpoint_model = os.path.join(checkpoint_dir, 'model_old.pth')

    from affordance_nets.models.main_models.trajectory.image_cond_film_models import ImageTrajectoryEBM as Model
    from affordance_nets.models.vision_backbone.resnet import ResNet18_Backbone

    vision_backbone = ResNet18_Backbone()
    model = Model(vision_backbone=vision_backbone, H=horizon).to(device)
    model.load_state_dict(torch.load(checkpoint_model))
    print('Loaded Model')

    ## Test the models segemntation maps ##
    main(model, horizon = horizon)