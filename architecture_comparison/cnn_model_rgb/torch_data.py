__all__ = ["FolderDataset"]

import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import pickle


class FolderDataset(Dataset):
    """
    Creates a PyTorch dataset from folder, returning two tensor images.
    Args:
    main_dir : directory where images are stored.
    img_dict_path: directory where the dictionary of the img is stored.
    transform (optional) : torchvision transforms to be applied while making dataset
    """

    def __init__(self, main_dir, main_dir_output, img_dict_path, transform_in=None, transform_out=None, tensor_dim=128):
        self.main_dir = main_dir
        self.main_dir_output = main_dir_output
        self.transform_in = transform_in
        self.transform_out = transform_out
        self.tensor_dim = tensor_dim

        # Filter files by type
        self.all_imgs = []
        for file in os.listdir(main_dir):
            # check only text files
            if file.lower().endswith(('.png', 'jpg')):
                self.all_imgs.append(file)

        # Output images:
        if main_dir != main_dir_output:
            self.all_imgs_output = []
            for file in os.listdir(main_dir_output):
                # check only text files
                if file.lower().endswith(('.png', 'jpg')):
                    self.all_imgs_output.append(file)
        else:
            self.all_imgs_output = self.all_imgs

        # Create img name dictionary
        self.dict = {}
        for idx, img in enumerate(self.all_imgs):
            self.dict[idx] = img

        with open(img_dict_path, 'wb') as f:
            pickle.dump(self.dict, f)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])

        image = Image.open(img_loc).convert("RGB")
        image_out = image.copy()

        size_pre = 132
        image.thumbnail((size_pre, size_pre))
        img_processed = Image.new('RGB', (size_pre, size_pre), (0, 0, 0))
        img_processed.paste(image, (int((size_pre - image.width) / 2), int((size_pre - image.height) / 2)))
        tensor_image_in = self.transform_in(img_processed)

        # Out image
        image_out.thumbnail((self.tensor_dim, self.tensor_dim))
        img_processed_out = Image.new('RGB',  (self.tensor_dim, self.tensor_dim), (0, 0, 0))
        img_processed_out.paste(image_out, (int((self.tensor_dim - image_out.width) / 2), int((self.tensor_dim - image_out.height) / 2)))
        tensor_image_out = self.transform_out(img_processed_out)

        return tensor_image_in, tensor_image_out
