__all__ = ["FolderDataset"]

import torch
from PIL import Image
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle
import kornia
import torchvision.transforms as T


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
        def create_tensor(directory, img_title, size):
            # Open image
            img_loc = os.path.join(directory, img_title)
            image = Image.open(img_loc).convert("RGB")

            # Transform image: Thumb to size, Center in black bg, transform
            image.thumbnail((size, size))
            img_processed = Image.new('RGB', (size, size), (0, 0, 0))
            img_processed.paste(image, (int((size - image.width) / 2), int((size - image.height) / 2)))
            tensor_rgb = self.transform_in(img_processed)

            # Edge detection
            edge_layer = kornia.filters.canny(tensor_rgb[None, :])[1].view(1, 128, 128)

            # RGB transform: Blur & Crop
            rgb_transforms = T.Compose([T.GaussianBlur(kernel_size=9, sigma=5),
                                        T.CenterCrop(size=(98, 78)),
                                        T.Pad((25, 15))])
            tensor_rgb_transformed = rgb_transforms(rgb_transforms(tensor_rgb))

            # Combine vectors
            tensor = torch.cat((tensor_rgb_transformed, edge_layer), 0)

            return tensor

        tensor_image_in = create_tensor(self.main_dir, self.all_imgs[idx], self.tensor_dim)
        tensor_image_out = create_tensor(self.main_dir_output, self.all_imgs_output[idx], self.tensor_dim)

        return tensor_image_in, tensor_image_out
