__all__ = [
    "load_image_tensor",
    "compute_similar_images",
    "plot_similar_images",
    "set_vars"
]

import torch
import numpy as np
import cnn_model.config as config
import cnn_model.torch_model as torch_model
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pickle
import kornia


def load_image_tensor(image_path, device, file_name="", angle_dict=None, clip_edge=True, normalize=True):
    image = Image.open(image_path)
    size = config.IMG_WIDTH
    image.thumbnail((size, size))
    img_processed = Image.new('RGB',  (size, size), (0, 0, 0))
    img_processed.paste(image, (int((size - image.width) / 2), int((size - image.height) / 2)))

    transforms = T.Compose([T.ToTensor()])
    tensor_rgb = transforms(img_processed)

    # Edge detection
    edge_layer = kornia.filters.canny(tensor_rgb[None, :])[1].view(1, 128, 128)

    # Clip side edges
    if clip_edge:
        crop_x = int((size - image.width) / 2) * 2 + 4 if image.width < size else 0
        crop_y = int((size - image.height) / 2) * 2 + 4 if image.height < size else 0

        crop = T.Compose([T.CenterCrop(size=(size - crop_y, size - crop_x)), T.Pad((int(crop_x / 2), int(crop_y / 2)))])
        edge_layer = crop(edge_layer)

    # Rotation
    if angle_dict is not None:
        angle = angle_dict.get(file_name)
        if angle is not None:
            edge_layer = T.functional.rotate(edge_layer, -int(angle))
            tensor_rgb = T.functional.rotate(tensor_rgb, -int(angle))

    # RGB transform: Blur & Crop
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

    # Unsqueze
    tensor = tensor.unsqueeze(0)

    # print(image_tensor.shape)
    # input_images = image_tensor.to(device)
    return tensor


def compute_similar_images(image_path, num_images, embedding, encoder, device, img_dict, file_name="", angle_dict=None):
    image_tensor = load_image_tensor(image_path, device, file_name, angle_dict)

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()

    # print(image_embedding.shape)

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    # print(flattened_embedding.shape)

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    distances, indices = knn.kneighbors(flattened_embedding)

    image_list = []
    for idx, indice in enumerate(indices[0]):
        if indice != 0:
            # index 0 is a dummy embedding.
            img_name = str(img_dict.get(indice - 1))
            image_list.append([img_name, float(distances[0][idx])])

    return image_list


def plot_similar_images(img_list):
    """
    Plots images that are similar to indices obtained from computing simliar images.
    Args:
    img_list : List of List of all images. E.g. [[1, 2, 3]]
    """

    for image in img_list:
            img_path = os.path.join(config.DATA_PATH + image[0])
            # print(img_path)
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.show()
            # img.save(f"../data/outputs/recommended_{index - 1}.jpg")


def plot_similar_images_grid(query, img_list, title='', sim_path=config.DATA_PATH, img_size=config.IMG_HEIGHT):
    # Size of the img based on the number of given images
    padding = int(img_size / 8)
    img_size_padded = int(img_size + 2 * padding)
    grid_size = int(np.ceil(len(img_list) / 5))
    grid_height = int((img_size_padded + padding) * grid_size + padding * 6) if grid_size > 1 else int(img_size_padded * 2 + padding * 6)
    grid_width = 7*img_size_padded + 2*padding

    # Create the background
    grid = Image.new('RGB',  (grid_width, grid_height), (255, 255, 255))

    # Prepare to draw text
    draw = ImageDraw.Draw(grid)
    def font(size):
        return ImageFont.truetype("Arial.ttf", size=int(size))

    # Add query image
    query_img_size = 2*img_size + 3*padding
    querry_img = Image.open(query)
    querry_img.thumbnail((query_img_size, query_img_size))
    grid.paste(querry_img, (padding, 5*padding))

    # Add titles
    draw.text((padding, padding), title, font=font(padding * 1.6), fill = (0, 0, 0))
    draw.text((padding, padding * 3), 'Query', font=font(padding*1.4), fill=(0, 0, 0))
    draw.text((query_img_size + 4 * padding, padding * 3), "Similar images", font=font(padding * 1.4), fill=(0, 0, 0))

    # Create list of positions
    positions = []
    for y in range(0, grid_size):
        for x in range(0, 5):
            pos_x = int(query_img_size + 4 * padding + x * img_size_padded)
            pox_y = int(5*padding + y * (img_size_padded + padding))
            positions.append((pos_x, pox_y))

    # Add the similar images to grid
    for idx, sim_img in enumerate(img_list):
        try:
            img = Image.open(sim_path+sim_img[0])
            img.thumbnail((img_size, img_size))
        except:
            print("An exception occurred: Image not found")
            img = Image.new('RGB',  (img_size, img_size), (180, 180, 180))
            size = int(img_size/10)
            ImageDraw.Draw(img).text((size, size), 'Image Loading Error', font=ImageFont.truetype("Arial.ttf", size=size), fill=(0, 0, 0))

        grid.paste(img, (positions[idx][0], positions[idx][1]))

        # Add title
        text_pos_y = int(positions[idx][1] + img_size + padding*0.6)
        img_title = sim_img[0] if len(sim_img[0]) < 20 else sim_img[0][:20]+"..."
        draw.text((positions[idx][0], text_pos_y), img_title, font=font(padding * 0.8), fill=(0, 0, 0))
        draw.text((positions[idx][0], text_pos_y + padding), str(round(sim_img[1],6)), font=font(padding*0.8), fill=(50, 50, 50))

    return grid


def set_vars(src_encoder=config.ENCODER_MODEL_PATH, src_dict=config.IMG_DICT_PATH, src_embedding=config.EMBEDDING_PATH):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    encoder = torch_model.ConvEncoder()

    # Load the state dict of encoder
    encoder.load_state_dict(torch.load(src_encoder, map_location=device))
    encoder.eval()
    encoder.to(device)

    # Load the img name dict:
    with open(src_dict, 'rb') as f:
        img_dict = pickle.load(f)

    # Loads the embedding
    embedding = np.load(src_embedding)

    return encoder, img_dict, embedding, device


def plot_similar_cnn(query, embedding, encoder, device, img_dict,
                     num = config.NUM_IMAGES, img="", angle_dict=None):
    image_list = compute_similar_images(query, num, embedding, encoder, device, img_dict, img, angle_dict)
    return plot_similar_images_grid(query, image_list, 'CNN-Model')


if __name__ == "__main__":
    encoder, img_dict, embedding, device = set_vars()

    plot_similar_cnn('../data/coa_edited/7807_edited.jpg', embedding, encoder, device, img_dict)

    print("Hello")

    # image_list = compute_similar_features(test_img_path, config.NUM_IMAGES, embedding)
    # plot_similar_images_grid(test_img_path, image_list, 'CNN-Model')

