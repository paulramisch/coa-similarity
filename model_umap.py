# PCA-Model zum Vergleich Suche ähnlicher Wappen in einem Datenset
# Implementierung angelehnt an: https://umap.scikit-tda.org/transform.html
# UMAP Erklärung: https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668
# https://pair-code.github.io/understanding-umap/

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
import umap.umap_ as umap
from helper_functions import process_img
import warnings
warnings.filterwarnings('ignore')

# Load data
np.random.seed(42)
coa_data = pd.read_csv('data/training-data_40x40_rgb.csv')
coa_data.head()

# Img size: Square root of the number of columns, minus the label column (RGB)
# If greyscale: size = int((coa_data.shape[1] - 1) ** (1/2))
size = int(((coa_data.shape[1] - 1) / 3) ** (1/2))

# Data prep
coa_data_img = coa_data.drop('0', axis=1) / 255
coa_data_labels = coa_data['0']

# UMAP
# For parameters: https://umap.scikit-tda.org/parameters.html
trans = umap.UMAP(n_neighbors=100, min_dist=0.3, n_components=30).fit(coa_data_img)


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
    coa = process_img(img, size) / 255

    # Reformat as DataFrame
    coa_df = pd.DataFrame([np.concatenate((['coa'], coa))]).drop(0, axis=1)

    # Extract features
    img_features = trans.transform(coa_df)

    # Cosine distance comparison
    similar_idx = [distance.euclidean(img_features[0], feat) for feat in trans.embedding_]

    # Add cos distance to data
    coa_data_sorted = coa_data
    coa_data_sorted['euc_distance'] = similar_idx
    coa_data_sorted = coa_data_sorted.sort_values('euc_distance')

    # Plot most similar images
    plot_most_similar(coa_data_sorted.drop(['0', 'euc_distance'], axis=1))

    return coa_data_sorted[['0', 'euc_distance']].head(10).to_numpy()


compare('data/test_data/-1_G A lion cr..jpg')
#compare('data/test_data/-1_G E chevron.jpg')
#compare('data/test_data/-1_O B lion rampant.jpg')
print("Ready")

# Test
# Umap is stochastic: Not always the same: https://github.com/lmcinnes/umap/issues/566
title = '-1_G A lion cr..jpg'
a = trans.transform(coa_data[coa_data['0'] == title].drop(['0', 'euc_distance'], axis=1))[0]
b = trans.transform(coa_data.drop(['0', 'euc_distance'], axis=1))[coa_data[coa_data['0'] == title].index.values[0]]
print(distance.euclidean(a, b))

# Distance: euclidean distance instead of cosine, because Umap is in euclidean space
# https://github.com/lmcinnes/umap/issues/519