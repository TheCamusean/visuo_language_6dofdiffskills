import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from affordance_nets.utils.directory_utils import get_data_dir



class RLBenchDataset1():
    def __int__(self, task_type='open_drawer', type='train'):
        data_path = os.path.join(get_data_dir(), 'rl_bench')
        self.data_path = os.path.join(data_path, type, task_type)

        self.episode_len = 100
        self.len = self.episode_len*10

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample_image, sample_mask = self.generate_2d_image_and_mask(self.poses[idx])
        return sample_image, sample_mask


class Julen():
    def __int__(self):
        print('hole')


if __name__ == '__main__':
    j = Julen()

    dataset = RLBenchDataset1()

    batch_size = 5
    my_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Example of iterating through the DataLoader
    for batch_data, batch_targets in my_dataloader:
        print(batch_data, batch_targets)