__all__ = ["FolderDataset"]

import torch
from PIL import Image
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle
import kornia
import torchvision.transforms as T
import csv


class FolderDataset(Dataset):
    """
    Creates a PyTorch dataset from folder, returning two tensor images.
    Args:
    main_dir : directory where images are stored.
    img_dict_path: directory where the dictionary of the img is stored.
    transform (optional) : torchvision transforms to be applied while making dataset
    """

    def __init__(self, main_dir, main_dir_output, img_dict_path,
                 transform_in, transform_in_after=None, transform_out=None,
                 tensor_dim=128, angle_dict_path=None):
        self.main_dir = main_dir
        self.main_dir_output = main_dir_output
        self.transform_in = transform_in
        self.transform_in_after = transform_in_after
        self.transform_out = transform_out
        self.tensor_dim = tensor_dim

        if angle_dict_path is not None:
            with open(angle_dict_path, mode='r') as infile:
                reader = csv.reader(infile)
                self.angle_dict = dict((rows[0], rows[1]) for rows in reader)

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
        def create_tensor(directory, img_title, size, noise=False, clip_edge=True, normalize=True):
            # Open image
            img_loc = os.path.join(directory, img_title)
            image = Image.open(img_loc).convert("RGB")

            # Resize image: Thumb to size
            image.thumbnail((size, size))

            # Create black background in size x size
            img_processed = Image.new('RGB', (size, size), (0, 0, 0))

            # Add image in the center of the new background
            img_processed.paste(image, (int((size - image.width) / 2), int((size - image.height) / 2)))

            # Add transforms, mostly random jitter, rotation etc.; depending on torch_train
            tensor_rgb = self.transform_in(img_processed)

            # Edge detection
            edge_layer = kornia.filters.canny(tensor_rgb[None, :])[1].view(1, 128, 128)

            # Clip side edges
            if clip_edge:
                crop_x = int((size - image.width) / 2)*2 + 4 if image.width < size else 0
                crop_y = int((size - image.height) / 2)*2 + 4 if image.height < size else 0

                crop = T.Compose([T.CenterCrop(size=(size - crop_y, size - crop_x)), T.Pad((int(crop_x/2), int(crop_y/2)))])
                edge_layer = crop(edge_layer)

            # Rotation etc.
            if self.transform_in_after:
                tensor_rgb = self.transform_in_after(tensor_rgb)
                edge_layer = self.transform_in_after(edge_layer)


            # Add rotation (if exists)
            if self.angle_dict is not None:
                angle = self.angle_dict.get(img_title)
                if angle is not None:
                    edge_layer = T.functional.rotate(edge_layer, -int(angle))
                    tensor_rgb = T.functional.rotate(tensor_rgb, -int(angle))

            # RGB transform: Blur & Crop
            # Todo: Crop & Pad values not hardcoded but through size
            rgb_transforms = T.Compose([T.GaussianBlur(kernel_size=7, sigma=5),
                                        T.CenterCrop(size=(98, 78)),
                                        T.Pad((25, 15))])
            tensor_transformed = rgb_transforms(rgb_transforms(tensor_rgb))

            # Normalize
            if normalize:
                mean, std = tensor_transformed.mean([1, 2]), tensor_transformed.std([1, 2])
                normalizer = T.Compose([T.Normalize(mean, std)])
                tensor_transformed = normalizer(tensor_transformed)

            # Combine vectors
            tensor = torch.cat((tensor_transformed, edge_layer), 0)

            # Add noise
            if noise:
                tensor = tensor + torch.randn(4, size, size) * (0.1**0.5)

            return tensor

        tensor_image_in = create_tensor(self.main_dir, self.all_imgs[idx], self.tensor_dim, noise=True)
        tensor_image_out = create_tensor(self.main_dir_output, self.all_imgs_output[idx], self.tensor_dim)

        return tensor_image_in, tensor_image_out
