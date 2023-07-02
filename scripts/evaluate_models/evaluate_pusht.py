import os
import time

import torch
from torch.utils.data import DataLoader
from affordance_nets.datasets.pusht_diffusion_policy_data import PushTImageProjectedDataset
from affordance_nets.models.segmentation_net import SegmentationNet as Model
from affordance_nets.utils.directory_utils import makedirs
import torch.optim as optim
from tqdm import tqdm


import matplotlib.pyplot as plt


# GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(model):

    from affordance_nets.utils.directory_utils import get_data_dir
    H=10
    batch_size = 1
    path = get_data_dir()
    zarr_path = os.path.join(path, 'diffusion_policy', 'pusht', 'pusht_cchi_v7_replay.zarr')
    train_data = PushTImageProjectedDataset(zarr_path=zarr_path, horizon=H)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        # Evaluation loop
        for i, batch in enumerate(train_dataloader):
            inputs = batch['obs']['image'][:, 0, ...].to(device)
            labels = batch['obs']['agent_pos_mask'].to(device)

            start_time = time.time()
            out = model(pixel_values = inputs)
            print('Evaluation Time: {}'.format(time.time()- start_time))


            ## Visualize the Visual energy Maps ##
            out = torch.sigmoid(out[0]).cpu().numpy()
            labels = labels.cpu().numpy()

            _, ax = plt.subplots(2, 6, figsize=(8, 4))

            _image = inputs.permute(0, 2, 3, 1).cpu().numpy()
            _image = (_image - _image.min()) / (_image.max() - _image.min())
            k=0
            ax[0,0].imshow(_image[k,...])

            for l in range(5):
                img_vis = 0.5 * _image + 0.5 * out[:, l+4, ..., None]
                ax[0, l+1].imshow(img_vis[k, ...])

                img_vis = 0.5 * _image + 0.5 * labels[:, l+4, ..., None]
                ax[1, l+1].imshow(img_vis[k, ...])

            plt.show()





if __name__ == '__main__':
    ## Save Torch Model ##
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..' , 'trained_models', 'pusht')
    makedirs(checkpoint_dir)
    checkpoint_model = os.path.join(checkpoint_dir, 'model.pth')

    H=10
    model = Model(output_channels=H).to(device)
    model.load_state_dict(torch.load(checkpoint_model))
    print('Loaded Model')

    ## Test the models segemntation maps ##
    main(model)