# PCA-Model zum Vergleich Suche ähnlicher Wappen in einem Datenset
# Implementierung angelehnt an: https://umap.scikit-tda.org/transform.html
# UMAP Erklärung: https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668

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
coa_data = pd.read_csv('data/pca_training_data.csv')
coa_data.head()

# Img size: Square root of the number of columns, minus the label column (RGB)
# If greyscale: size = int((coa_data.shape[1] - 1) ** (1/2))
size = int(((coa_data.shape[1] - 1) / 3) ** (1/2))

# Data prep
coa_data_img = coa_data.drop('0', axis=1)
coa_data_labels = coa_data['0']

# UMAP
# For parameters: https://umap.scikit-tda.org/parameters.html
trans = umap.UMAP(n_neighbors=3, random_state=42, n_components=50).fit(coa_data_img)


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
    img_features = trans.transform(coa_df)

    # Cosine distance comparison
    similar_idx = [ distance.cosine(img_features[0], feat) for feat in trans.embedding_ ]

    # Add cos distance to data
    coa_data_sorted = coa_data
    coa_data_sorted['cos_distance'] = similar_idx
    coa_data_sorted = coa_data_sorted.sort_values('cos_distance')

    # Plot most similar images
    plot_most_similar(coa_data_sorted.drop(['0', 'cos_distance'], axis=1))

    return coa_data_sorted[['0', 'cos_distance']].head(10).to_numpy()


compare('data/test_data/1.png')
print("Ready")