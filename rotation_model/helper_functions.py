from PIL import Image, ImageOps
import os
import torchvision.transforms as T
import torch
import kornia


def create_tensor(directory, img_title, size, edge_detection, mirror=False, normalize=False):
    if type(img_title) == tuple:
        img_mirror = img_title[1]
        img_title = img_title[0]
    else:
        img_mirror = False

    # Open image
    img_loc = os.path.join(directory, img_title)
    image = Image.open(img_loc).convert("RGB")

    # Mirror images
    if mirror & img_mirror:
        image = ImageOps.mirror(image)

    # Transform image: Thumb to size, Center in black bg, transform
    image.thumbnail((size, size))
    img_processed = Image.new('RGB', (size, size), (0, 0, 0))
    img_processed.paste(image, (int((size - image.width) / 2), int((size - image.height) / 2)))
    transforms = T.Compose([T.ToTensor()])
    tensor = transforms(img_processed)

    # Normalize
    if normalize:
        mean, std = tensor.mean([1, 2]), tensor.std([1, 2])
        normalizer = T.Compose([T.Normalize(mean, std)])
        tensor = normalizer(tensor)

    # edge_detection
    if edge_detection:
        # tensor[None, :] & .view(1, 128, 128) because of batch management (it wants 1 more dimension)
        edges = kornia.filters.canny(tensor[None, :])
        edge_magnitudes_map = edges[0].view(1, 128, 128)
        edge_layer = edges[1].view(1, 128, 128)

        tensor = torch.cat((tensor, edge_magnitudes_map, edge_layer), 0)

    return tensor


def predict_angle(model, img_path, img, size=128, edge_detection=False):
    img_tensor = create_tensor(img_path, img, size, edge_detection)[None, :]

    with torch.no_grad():
        prediction = model(img_tensor).cpu().detach().numpy()

    prediction = prediction[0] * 90

    return prediction
