__all__ = ["FolderDataset"]

import os
from torch.utils.data import Dataset
import helper_functions
import csv


# Creates a PyTorch dataset from folder, returning two tensor images.
class FolderDataset(Dataset):

    def __init__(self, img_path, angle_data_path, tensor_dim=128):
        self.img_path = img_path
        self.tensor_dim = tensor_dim

        # Set data dictionary
        with open(angle_data_path, mode='r') as infile:
            reader = csv.reader(infile)
            self.angle_dict = dict((rows[0],rows[1]) for rows in reader)

        # Filter files by type
        self.all_imgs = []
        for file in os.listdir(img_path):
            # check only text files
            if file.lower().endswith(('.png', 'jpg')):
                self.all_imgs.append(file)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        tensor_image = helper_functions.create_tensor(self.img_path, self.all_imgs[idx], self.tensor_dim)
        angle = int(self.angle_dict.get(self.all_imgs[idx], 0))/ 90

        return tensor_image, angle
