import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os


data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
print(data_path)

class SimpleNaiveData(Dataset):

    def __init__(self):
        dir_path = os.path.join(data_path, 'naive_data')

        pose_path = os.path.join(dir_path, 'poses.npy')
        self.poses = np.load(pose_path)
        self.len = self.poses.shape[0]

    def __len__(self):
        return self.len

    def generate_2d_image_and_mask(self, idx):
        import matplotlib.pyplot as plt
        import numpy as np

        # Create a 224x224x3 array of zeros (an empty RGB image)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        mask = np.zeros((224, 224), dtype=np.uint8)

        # Define the size of the square
        square_size = 40

        # Generate random coordinates for the top left corner of the square
        x = int(idx[0])
        y = int(idx[1])
        if x<224 and y<224:
            # Change the color of the pixels in the square to blue
            image[x:x + square_size, y:y + square_size] = [0, 0, 255]

            ## Set Segmentation mask ##
            mask[x:x + square_size, y:y + square_size] = 1

        # # Display the image
        # plt.imshow(mask)
        # plt.axis('off')  # to remove the axis
        # plt.show()
        return image, mask

    def __getitem__(self, idx):
        sample_image, sample_mask = self.generate_2d_image_and_mask(self.poses[idx])
        return sample_image, sample_mask




if __name__ == '__main__':
    data = SimpleNaiveData()

    batch_size = 5
    my_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # Example of iterating through the DataLoader
    for batch_data, batch_targets in my_dataloader:
        print(batch_data, batch_targets)

