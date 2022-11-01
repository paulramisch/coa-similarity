__all__ = ["FolderDataset"]

import torch
from PIL import Image
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle


class FolderDataset(Dataset):
    """
    Creates a PyTorch dataset from folder, returning two tensor images.
    Args:
    main_dir : directory where images are stored.
    IMG_DICT_PATH: directory where the dictionary of the img is stored.
    transform (optional) : torchvision transforms to be applied while making dataset
    """

    def __init__(self, main_dir, IMG_DICT_PATH, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)

        # Create img name dictionary
        self.dict = {}
        for idx, img in enumerate(self.all_imgs):
            self.dict[idx] = img

        with open(IMG_DICT_PATH, 'wb') as f:
            pickle.dump(self.dict, f)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image, tensor_image
