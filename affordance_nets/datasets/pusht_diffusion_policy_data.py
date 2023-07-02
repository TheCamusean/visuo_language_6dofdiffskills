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



if __name__ == '__main__':
    import os
    from affordance_nets.utils.directory_utils import get_data_dir

    path = get_data_dir()
    zarr_path = os.path.join(path, 'diffusion_policy', 'pusht', 'pusht_cchi_v7_replay.zarr')

    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = PushTImageDataset(
        zarr_path=zarr_path,
        horizon=pred_horizon,
        pad_before = 1,
        pad_after = 8,
    )


    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=5,
        # num_workers=1,
        # shuffle=True,
        # # accelerate cpu-gpu transfer
        # pin_memory=True,
        # # don't kill worker process afte each epoch
        # persistent_workers=True
    )

    # visualize data in batch
    batch = next(iter(dataloader))
    print(batch)