# Funktionen zur Findung ähnlicher Wappen

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
import umap.umap_ as umap
from helper_functions import process_img_pca
import warnings
warnings.filterwarnings('ignore')


# Load and Transform image
def load_img(img_src):
    # Load image
    img = cv2.imread(img_src)

    # Transform image
    processed_img = process_img_pca(img, 100)

    return processed_img


# Sort data by cosinus distance
def sort_data(data, coa_sim):
    data_sorted = data
    data_sorted['cos_distance'] = coa_sim

    return data_sorted.sort_values('cos_distance')


# Function to plot most similar images
def plot_most_similar(pixels):
    fig, axes = plt.subplots(1, 5, figsize=(6, 2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(size, size, 3))
    plt.show()

# PCA
def compare_umap(model, img_src, coa_data):
    coa = load_img(img_src)

    # Reformat as DataFrame
    coa_df = pd.DataFrame([np.concatenate((['coa'], coa))]).drop(0, axis=1)

    # Extract features
    img_features = create_feature(coa_df)

    # Cosine distance comparison
    similar_idx = [ distance.cosine(img_features[0], feat) for feat in pca.transform(images) ]

    # Add cos distance to data
    coa_data_sorted = sort_data(coa_data, similar_idx)

    # Plot most similar images
    plot_most_similar(coa_data_sorted.drop(['0', 'cos_distance'], axis=1))

    return coa_data_sorted[['0', 'cos_distance']].head(10).to_numpy()

# Umap
def compare_umap(model, img_src, coa_data):
    coa = load_img(img_src)

    # Reformat as DataFrame
    coa_df = pd.DataFrame([np.concatenate((['coa'], coa))]).drop(0, axis=1)

    # Extract features
    img_features = model.transform(coa_df)

    # Cosine distance comparison
    similar_idx = [ distance.cosine(img_features[0], feat) for feat in model.embedding_ ]

    # Add cos distance to data
    coa_data_sorted = sort_data(coa_data, similar_idx)

    # Plot most similar images
    plot_most_similar(coa_data_sorted.drop(['0', 'cos_distance'], axis=1))

    return coa_data_sorted[['0', 'cos_distance']].head(10).to_numpy()

