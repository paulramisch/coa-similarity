__all__ = ["FolderDataset"]

import os
from torch.utils.data import Dataset
import helper_functions
import csv


# Creates a PyTorch dataset from folder, returning two tensor images.
class FolderDataset(Dataset):

    def __init__(self, img_path, angle_data_path, tensor_dim=128, edge_detection=False, mirror=False):
        self.img_path = img_path
        self.tensor_dim = tensor_dim

        # Set data dictionary
        with open(angle_data_path, mode='r') as infile:
            reader = csv.reader(infile)
            self.angle_dict = dict((rows[0],rows[1]) for rows in reader)

        # Filter files by type
        self.all_imgs = []
        for file in os.listdir(img_path):
            # check only img files
            if file.lower().endswith(('.png', 'jpg')):
                if mirror:
                    self.all_imgs.append((file, True))
                    self.all_imgs.append((file, False))
                else:
                    self.all_imgs.append(file)


        # Use edge_detection
        self.edge_detection = edge_detection

        # Use mirror input img
        self.mirror = mirror

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        tensor_image = helper_functions.create_tensor(self.img_path, self.all_imgs[idx], self.tensor_dim,
                                                      self.edge_detection, self.mirror)
        if self.mirror:
            sign = -1 if self.all_imgs[idx][1] else 1
            angle = sign * int(self.angle_dict.get(self.all_imgs[idx][0], 0))/ 90
        else:
            angle = int(self.angle_dict.get(self.all_imgs[idx], 0)) / 90

        return tensor_image, angle
