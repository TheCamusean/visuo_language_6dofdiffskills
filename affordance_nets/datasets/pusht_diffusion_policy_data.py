from typing import Dict
import torch
import numpy as np
import copy

from torch.utils.data import DataLoader, Dataset

from affordance_nets.common.pytorch_util import dict_apply
from affordance_nets.common.replay_buffer import ReplayBuffer
from affordance_nets.common.sampler import (
     SequenceSampler, get_val_mask, downsample_mask)
# from diffusion_policy.model.common.normalizer import LinearNormalizer
# from diffusion_policy.dataset.base_dataset import BaseImageDataset
# from diffusion_policy.common.normalize_util import get_image_range_normalizer
import kornia

class PushTImageDataset(Dataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None
                 ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][..., :2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:, :2].astype(np.float32)  # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'], -1, 1) / 255

        data = {
            'obs': {
                'image': image,  # T, 3, 96, 96
                'agent_pos': agent_pos,  # T, 2
            },
            'action': sample['action'].astype(np.float32)  # T, 2
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


class PushTImageProjectedDataset(Dataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 resize = None,
                 ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.resize = resize
        if resize is not None:
            self.kernel_size = (self.resize//(4*2))*2 +1
            self.sigma = self.resize//8
        else:
            self.kernel_size = 10
            self.sigma = 2

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][..., :2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def resize_idx(self, x, in_dim, out_dim):
        y = x/in_dim*out_dim
        return y

    def resize_image(self, img, in_dim, out_dim):
        x = np.arange(0, out_dim, 1)
        map_in = self.resize_idx(x, out_dim, in_dim)
        map_in = map_in.astype(int)

        img_out = np.zeros((img.shape[0], out_dim, out_dim, img.shape[-1]))
        xx, yy = np.meshgrid(x, x)
        x = xx.reshape(-1).astype(int)
        y = yy.reshape(-1).astype(int)
        img_out[:, x, y, :] = img[:, map_in[x], map_in[y], :]
        return img_out

    def _sample_to_data(self, sample):

        image = sample['img']
        ## Reshape Images for the Net ##
        if self.resize is not None:
            image = self.resize_image(image, image.shape[1], self.resize)

        img_size = image.shape[1]
        image_pixel_lenght = img_size

        ## Generate a mask of agent_pos ##
        agent_pos = sample['state'][:, :2]/512
        agent_pos_ = agent_pos*image_pixel_lenght
        t_lenght = agent_pos_.shape[0]

        agent_pos_mask = torch.zeros((t_lenght, image_pixel_lenght, image_pixel_lenght))
        agent_pos_mask[np.arange(0,agent_pos_.shape[0],1), agent_pos_[:,1].astype(int), agent_pos_[:,0].astype(int)] = 1


        agent_pos_mask = kornia.filters.gaussian_blur2d(agent_pos_mask[:,None,...], (self.kernel_size, self.kernel_size), (self.sigma, self.sigma)).squeeze()
        max_img = agent_pos_mask.reshape(agent_pos_mask.shape[0], -1).max(dim=1).values
        agent_pos_mask = agent_pos_mask/max_img[:,None,None]

        ## Generate a mask of agent_actions ##
        agent_act_p = sample['action']/512
        agent_act_p_ = agent_act_p*image_pixel_lenght

        agent_act_mask = np.zeros((t_lenght, image_pixel_lenght, image_pixel_lenght), dtype=np.uint8)
        agent_act_mask[np.arange(0,agent_act_p_.shape[0],1), agent_act_p_[:,1].astype(int), agent_act_p_[:,0].astype(int)] = 1

        VISUALIZE_IMG = False
        if VISUALIZE_IMG:

            agent_pos_mask = agent_pos_mask.numpy()
            image_2 = image.permute(0,2,3,1).numpy()

            n=5
            if n==0:
                img_vis = 0.5 * (img[:,...] / 255) + 0.5 * agent_pos_mask[:,...,None]
                img_vis2 = 0.5 * ((image_2 - image_2.min())/(image_2.max() - image_2.min())) + 0.5 * agent_pos_mask[:,...,None]
            else:
                img_vis = 0.5 * (img[:-n,...] / 255) + 0.5 * agent_pos_mask[n:,...,None]
                img_vis2 = 0.5 * (image_2[:-n,...]) + 0.5 * agent_pos_mask[n:,...,None]


            #for k in range(img_vis.shape[0]):
            k=0
            _, ax = plt.subplots(1, 3, figsize=(8, 4))

            ax[0].imshow(img_vis[k, ...])
            ax[1].imshow(agent_pos_mask[k, ...])
            ax[2].imshow(img_vis2[k, ...])

            plt.show()

        data = {
            'obs': {
                'image': image,  # T, 3, 96, 96
                'agent_pos':  agent_pos,  # T, 2
                'agent_pos_mask': agent_pos_mask,  # T, 96, 96
            },
            'action':{
                 'action': agent_act_p,  # T, 2
                 'action_mask': agent_act_mask,  # T, 96,96
            }
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data



if __name__ == '__main__':
    import os
    from affordance_nets.utils.directory_utils import get_data_dir

    path = get_data_dir()
    zarr_path = os.path.join(path, 'diffusion_policy', 'pusht', 'pusht_cchi_v7_replay.zarr')

    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8


    PROJECTED_IMAGE = False
    if PROJECTED_IMAGE:

        ########### Projected Image #################
        dataset = PushTImageProjectedDataset(
            zarr_path=zarr_path,
            horizon=pred_horizon,
            pad_before=0,
            pad_after=0,
            resize=200,
        )

        dataloader = torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=False)

        import matplotlib.pyplot as plt

        for i, batch in enumerate(dataloader):
            print('here')

        ##############################################


    ######################################################

    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = PushTImageProjectedDataset(
        zarr_path=zarr_path,
        horizon=pred_horizon,
        pad_before = 0,
        pad_after = 0,
        resize=200,
    )


    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )

    import matplotlib.pyplot as plt

    for i, batch in enumerate(dataloader):

        k = 0
        img = batch['obs']['image'][0,k,...].numpy()/255
        agent_poses = batch['obs']['agent_pos'].numpy()[0,...]*(img.shape[1])
        agent_pos = agent_poses[k,...]

        agent_actions = batch['action']['action'].numpy()[0,...]*(img.shape[1])
        agent_act = agent_actions[k, ...]

        img2 = img.copy()
        img2[int(agent_pos[1]), int(agent_pos[0]),:] = [0, 0, 0]

        H = 10
        # for l in range(1, H):
        #     agent_pos_pred = agent_poses[l, :2]
        #     img2[int(agent_pos_pred[1]), int(agent_pos_pred[0])] = [0, 255, 0]

        # for l in range(1, 1):
        #     agent_pos_pred = agent_actions[l, :2]
        #     img2[int(agent_pos_pred[1]), int(agent_pos_pred[0])] = [255, 0, 0]


        img2[int(agent_act[1]), int(agent_act[0])] = [1, 0, 0]

        plt.imshow(img2)
        plt.show()


