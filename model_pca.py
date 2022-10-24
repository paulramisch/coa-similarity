# PCA-Model zum Vergleich Suche ähnlicher Wappen in einem Datenset

# Implementierung angelehnt an:
# https://towardsdatascience.com/eigenfaces-face-classification-in-python-7b8d2af3d3ea
# https://github.com/ml4a/ml4a/blob/master/examples/info_retrieval/image-search.ipynb

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance
from helper_functions import process_img
import warnings
warnings.filterwarnings('ignore')

# Load data
coa_data = pd.read_csv('data/pca_training_data.csv')

# Img size: Square root of the number of columns, minus the label column (RGB)
# If greyscale: size = int((coa_data.shape[1] - 1) ** (1/2))
size = int(((coa_data.shape[1] - 1) / 3) ** (1/2))

# Data prep
coa_data_img = coa_data.drop('0', axis=1)
coa_data_labels = coa_data['0']

# PCA
# pca = PCA().fit(coa_data_img)

# Visual analysis of the number of needed variables
# plt.figure(figsize=(18, 7))
# plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3)
# plt.show()
# len(np.where(pca.explained_variance_ratio_.cumsum() < 0.99)[0])

# Training with component number based on analysis
pca = PCA(n_components=20).fit(coa_data_img)


# Transform images into feature vectors
def create_feature(images):
    return pca.transform(images)


pca_features = create_feature(coa_data_img)


# Function to compare
def compare(img_src):
    # Function to plot most similiar images
    def plot_most_similar(pixels):
        fig, axes = plt.subplots(1, 5, figsize=(6, 2))
        for i, ax in enumerate(axes.flat):
            ax.imshow(np.array(pixels)[i].reshape(size, size, 3))
        plt.show()

    # Load image
    img = cv2.imread(img_src)

    # Transform image
    coa = process_img(img, 100)

    # Reformat as DataFrame
    coa_df = pd.DataFrame([np.concatenate((['coa'], coa))]).drop(0, axis=1)

    # Extract features
    img_features = create_feature(coa_df)

    # Cosine distance comparison
    similar_idx = [ distance.cosine(img_features[0], feat) for feat in pca_features ]

    # Add cos distance to data
    coa_data_sorted = coa_data
    coa_data_sorted['cos_distance'] = similar_idx
    coa_data_sorted = coa_data_sorted.sort_values('cos_distance')

    # Plot most similar images
    plot_most_similar(coa_data_sorted.drop(['0', 'cos_distance'], axis=1))

    return coa_data_sorted[['0', 'cos_distance']].head(10).to_numpy()

compare('data/test_data/1.png')