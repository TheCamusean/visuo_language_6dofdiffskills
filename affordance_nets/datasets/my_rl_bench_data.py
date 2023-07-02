import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
import glob
import pickle

from affordance_nets.utils.directory_utils import get_data_dir

class RLBenchDataset(Dataset):

    def __init__(self, task_type='open_drawer', type='train'):
        data_path = os.path.join(get_data_dir(), 'rl_bench')
        self.data_path = os.path.join(data_path, type, task_type)

        ## Episode List ##
        self.episodes_folders = glob.glob(os.path.join(self.data_path, 'all_variations', 'episodes','**'))
        self.episode_len = len(self.episodes_folders)

        file_path = os.path.join(self.episodes_folders[0], 'low_dim_obs.pkl')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(data)

        self.len = self.episode_len*10


    def __len__(self):
        return self.len

    def generate_samples(self, idx):
        file_path = os.path.join(self.episodes_folders[idx], 'low_dim_obs.pkl')
        with open(file_path, 'rb') as f:
            low_dim_data = pickle.load(f)

        time = np.random.randint(0, len(low_dim_data._observations))
        low_dim_data_t = low_dim_data._observations[time]
        print(low_dim_data)

        H_gripper = low_dim_data_t.gripper_matrix

        misc_dict = low_dim_data_t.misc

    def __getitem__(self, idx):
        episode_idx = np.random.randint(0, self.episode_len)

        sample_image, sample_mask = self.generate_samples(episode_idx)
        return sample_image, sample_mask




if __name__ == '__main__':
    data = RLBenchDataset()

    batch_size = 5
    my_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # Example of iterating through the DataLoader
    for batch_data, batch_targets in my_dataloader:
        print(batch_data, batch_targets)

