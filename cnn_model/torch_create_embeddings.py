# Create/Update embeddings with existing model
# This script create an embeddings file and a dict file based on the path in model_path and img in img_path

import torch
import cnn_model.torch_model as torch_model
import cnn_model.torch_engine as torch_engine
import cnn_model.torch_data as torch_data
import torchvision.transforms as T
import cnn_model.config as config
import numpy as np


# Create embeddings
def create_embeddings(img_path, encoder, embedding_path, dict_path, shape, device):
    transforms = T.Compose([T.ToTensor()])

    full_dataset = torch_data.FolderDataset(img_path, config.IMG_PATH_OUTPUT, dict_path, transforms, transforms)
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=config.FULL_BATCH_SIZE)

    embedding = torch_engine.create_embedding(encoder, full_loader, shape, device)

    # Convert embedding to numpy and save them
    numpy_embedding = embedding.cpu().detach().numpy()
    num_images = numpy_embedding.shape[0]

    # Dump the embeddings for complete dataset, not just train
    flattened_embedding = numpy_embedding.reshape((num_images, -1))
    np.save(embedding_path, flattened_embedding)


# Create embeddings
if __name__ == "__main__":
    # Set model
    model_name = "_transformed3"
    model_path = "../data/models/"

    # Set devicee
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Set model paths
    img_path = "../data/coa_renamed/".format(model_name)
    encoder_path = "{}deep_encoder{}.pt".format(model_path, model_name)
    embedding_path = "{}data_embedding_f{}.npy".format(model_path, model_name)
    dict_path = "{}img_dict{}.pkl".format(model_path, model_name)
    shape = config.EMBEDDING_SHAPE

    # Load encoder
    encoder = torch_model.ConvEncoder()
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()
    encoder.to(device)

    # Create embeddings
    create_embeddings(img_path, encoder, embedding_path, dict_path, shape, device)
    print("Embeddings successfully created")