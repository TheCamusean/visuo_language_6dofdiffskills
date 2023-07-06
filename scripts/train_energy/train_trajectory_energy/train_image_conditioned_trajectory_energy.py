import os

import torch
from torch.utils.data import DataLoader
from affordance_nets.datasets.pusht_diffusion_policy_data import PushTImageProjectedDataset


from affordance_nets.utils.directory_utils import makedirs
import torch.optim as optim
from diffusers.optimization import get_scheduler

from tqdm import tqdm

import matplotlib.pyplot as plt
import wandb

wandb.login()

# Parameters
num_epochs = 1000
batch_size = 5
num_accumulation = 10
learning_rate = 1e-4

# GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    ## Set Model ##
    from affordance_nets.models.vision_backbone.resnet import ResNet18_Backbone
    from affordance_nets.models.main_models.trajectory.image_cond_film_models import ImageTrajectoryEBM as Model

    vision_backbone = ResNet18_Backbone()
    model = Model(vision_backbone=vision_backbone, H=10).to(device)

    load_pretrained = False
    if load_pretrained:
        ## Load Pretrained ##
        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'trained_models', 'pusht')
        checkpoint_model = os.path.join(checkpoint_dir, 'model_old.pth')
        model.load_state_dict(torch.load(checkpoint_model))

    ## Set Dataset
    from affordance_nets.utils.directory_utils import get_data_dir
    path = get_data_dir()
    zarr_path = os.path.join(path, 'diffusion_policy', 'pusht', 'pusht_cchi_v7_replay.zarr')

    train_data = PushTImageProjectedDataset(zarr_path=zarr_path, horizon=H, resize=vision_backbone.img_size)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # validation_data = PushTImageProjectedDataset(zarr_path=zarr_path, horizon=H, resize=vision_backbone.img_size)
    # val_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_dataloader) * num_epochs
    )


    wandb.init()
    # Training loop
    step = 0
    step_vis = 0
    NUM_ACCUMULATION_STEPS = num_accumulation
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        epoch_loss = 0
        # Wrap the dataloader with tqdm to show a progress bar
        for i, batch in enumerate(train_dataloader):
                step +=1

                images = batch['obs']['image'][:,0,...]
                x = batch['obs']['agent_pos'][:,:,:].to(device)

                # Forward pass
                # 1. Adapt Images
                images = model.vision_model.image_preprocess(images).to(device)


                context = {'images': images}
                outputs = model.train(x=x, context=context)
                loss = outputs['loss']

                # Backward pass and optimization
                # Normalize the Gradients
                loss_n = loss / NUM_ACCUMULATION_STEPS
                loss_n.backward()

                if ((step + 1) % NUM_ACCUMULATION_STEPS == 0):
                    step_vis += 1

                    # Update Optimizer
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()


                    if (step_vis + 1) % 5 == 0:
                        wandb.log({"epoch": epoch, "loss": loss}, step=step_vis)
                        print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item()}")


                    if (step_vis + 1)% 50 ==0:
                        ## Save Torch Model ##
                        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'trained_models', 'pusht')
                        makedirs(checkpoint_dir)
                        checkpoint_model = os.path.join(checkpoint_dir, 'model_old.pth')
                        torch.save(model.state_dict(), checkpoint_model)


    ## Save Torch Model ##
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'trained_models', 'naive_data')
    makedirs(checkpoint_dir)
    checkpoint_model = os.path.join(checkpoint_dir, 'model_old.pth')
    torch.save(model.state_dict(),  checkpoint_model)
    print('Finished Training')



if __name__ == '__main__':
    main()