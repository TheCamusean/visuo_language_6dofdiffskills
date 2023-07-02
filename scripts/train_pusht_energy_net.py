import os

import torch
from torch.utils.data import DataLoader
from affordance_nets.datasets.pusht_diffusion_policy_data import PushTImageProjectedDataset
from affordance_nets.models.segmentation_net import SegmentationNet as Model
from affordance_nets.utils.directory_utils import makedirs
import torch.optim as optim
from tqdm import tqdm


import matplotlib.pyplot as plt

# Parameters
num_epochs = 10
batch_size = 10
learning_rate = 0.0001

# GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'


def segmentation_visualization(img, mask):
    img = img.detach().cpu().numpy()
    mask = mask[...,None].repeat(1,1,1,3).detach().cpu().numpy()

    img_vis = 0.5*(img/255) + 0.5*mask

    for k in range(img_vis.shape[0]):
        _, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].imshow(img_vis[k,...])
        ax[1].imshow(mask[k,...])
        plt.show()



def main():

    ## Set Dataset
    H = 10

    from affordance_nets.utils.directory_utils import get_data_dir
    path = get_data_dir()
    zarr_path = os.path.join(path, 'diffusion_policy', 'pusht', 'pusht_cchi_v7_replay.zarr')

    train_data = PushTImageProjectedDataset(zarr_path=zarr_path, horizon=H)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    ## Set Model
    model = Model(output_channels=H).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        # Wrap the dataloader with tqdm to show a progress bar
        for i, batch in enumerate(train_dataloader):

                inputs = batch['obs']['image'][:,0,...].to(device)
                labels = batch['obs']['agent_pos_mask'].to(device)
                #labels = batch['obs']['agent_pos'].to(device)

                # Forward pass
                outputs = model(pixel_values=inputs, labels=labels)
                loss = outputs[0]

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 5 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item()}")


                if (i + 1)% 50 ==0:
                    ## Save Torch Model ##
                    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'trained_models', 'pusht')
                    makedirs(checkpoint_dir)
                    checkpoint_model = os.path.join(checkpoint_dir, 'model.pth')
                    torch.save(model.state_dict(), checkpoint_model)


    ## Save Torch Model ##
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'trained_models', 'naive_data')
    makedirs(checkpoint_dir)
    checkpoint_model = os.path.join(checkpoint_dir, 'model.pth')
    torch.save(model.state_dict(),  checkpoint_model)
    print('Finished Training')


    ############# Evaluation of the model ###############
    img = batch[0]
    labels = batch[1].to(device)
    outputs = model(**inputs, labels=labels)
    mask = torch.sigmoid(outputs[1])
    segmentation_visualization(img, mask)





if __name__ == '__main__':
    main()




