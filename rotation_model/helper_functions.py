from PIL import Image
import os
import torchvision.transforms as T
import torch


def create_tensor(directory, img_title, size):
    # Open image
    img_loc = os.path.join(directory, img_title)
    image = Image.open(img_loc).convert("RGB")

    # Transform image: Thumb to size, Center in black bg, transform
    image.thumbnail((size, size))
    img_processed = Image.new('RGB', (size, size), (0, 0, 0))
    img_processed.paste(image, (int((size - image.width) / 2), int((size - image.height) / 2)))
    transforms = T.Compose([T.ToTensor()])
    tensor = transforms(img_processed)

    # Normalize
    # mean, std = tensor.mean([1, 2]), tensor.std([1, 2])
    # normalizer = T.Compose([T.Normalize(mean, std)])
    # tensor = normalizer(tensor)

    return tensor


def predict_angle(model, img_path, img, size=128):
    img_tensor = create_tensor(img_path, img, size)[None, :]

    with torch.no_grad():
        prediction = model(img_tensor).cpu().detach().numpy()

    prediction = prediction[0] * 90

    return prediction
