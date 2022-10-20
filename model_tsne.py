# PCA-Model zum Vergleich Suche ähnlicher Wappen in einem Datenset

# Implementierung angelehnt an:
# https://towardsdatascience.com/eigenfaces-face-classification-in-python-7b8d2af3d3ea
# https://github.com/ml4a/ml4a/blob/master/examples/info_retrieval/image-search.ipynb

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
from helper_functions import process_img_pca
import warnings
warnings.filterwarnings('ignore')

# Load data
coa_data = pd.read_csv('data/pca_training_data.csv')
coa_data.head()


# Img size: Square root of the number of columns, minus the label column (RGB)
size = int(((coa_data.shape[1] - 1) / 3) ** (1/2))
# If greyscale:
# size = int((coa_data.shape[1] - 1) ** (1/2))


# Data prep
coa_data_img = coa_data.drop('0', axis=1)
coa_data_labels = coa_data['0']

# TSNE
# https://github.com/ml4a/ml4a/blob/master/examples/info_retrieval/image-tsne.ipynb
# Max 3 dimensions
tsne = TSNE(n_components=3, learning_rate=150, perplexity=30, angle=0.2, verbose=2)
tsne_result = tsne.fit_transform(coa_data_img)

# Function to compare
def compare(img):
    # Function to plot most similiar images


    def plot_most_similar(pixels):
        fig, axes = plt.subplots(1, 5, figsize=(6, 2))
        for i, ax in enumerate(axes.flat):
            ax.imshow(np.array(pixels)[i].reshape(size, size, 3))
        plt.show()

    # Cosine distance comparison
    similar_idx = [ distance.cosine(tsne_result[img], feat) for feat in tsne_result ]

    # Get most similar images
    idx_closest = sorted(range(len(similar_idx)), key=lambda k: similar_idx[k])[0:5]

    # Plot most similar images
    plot_most_similar(coa_data[coa_data.index.isin(idx_closest)].drop('0', axis=1))

    return coa_data[coa_data.index.isin(idx_closest)]['0'].values

compare(0)