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
from helper_functions import process_img, plot_similar_images_grid
import warnings
import csv
warnings.filterwarnings('ignore')

# Transform images into feature vectors
def create_feature(images, pca):
    return pca.transform(images)


def train_pca_model(coa_data, components_all=False, components=20):
    # Img size: Square root of the number of columns, minus the label column (RGB)
    # If greyscale: size = int((coa_data.shape[1] - 1) ** (1/2))
    size = int(((coa_data.shape[1] - 1) / 3) ** (1/2))

    # Data prep
    coa_data_img = coa_data.copy().drop('0', axis=1)
    coa_data_labels = coa_data[['0']].copy()

    # PCA
    if components_all:
        pca = PCA().fit(coa_data_img)
    else:
        # Training with component number based on analysis
        pca = PCA(n_components=components).fit(coa_data_img)

    pca_features = create_feature(coa_data_img, pca)
    return pca, pca_features, size


# Easy function to plot most similar images
def plot_most_similar(pixels):
    fig, axes = plt.subplots(1, 5, figsize=(6, 2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(size, size, 3))
    plt.show()


# Function to plot the grid view
def plot_similar_pca(query, coa_data, pca, pca_features, size):
    image_list = compare(query, coa_data, pca, pca_features, size)
    return plot_similar_images_grid(query, image_list, 'PCA-Model')


# Function to compare
def compare(img_src, coa_data, pca, pca_features, size):
    # Load image
    img = cv2.imread(img_src)

    # Transform image
    coa = process_img(img, size)

    # Reformat as DataFrame
    coa_df = pd.DataFrame([np.concatenate((['coa'], coa))]).drop(0, axis=1)

    # Extract features
    img_features = create_feature(coa_df, pca)

    # Cosine distance comparison
    similar_idx = [ distance.cosine(img_features[0], feat) for feat in pca_features ]

    # Add cos distance to data
    coa_data_sorted = coa_data[['0']].copy()
    coa_data_sorted['cos_distance'] = similar_idx
    coa_data_sorted = coa_data_sorted.sort_values('cos_distance')

    # Plot most similar images
    # plot_most_similar(coa_data_sorted.drop(['0', 'cos_distance'], axis=1))

    return coa_data_sorted[['0', 'cos_distance']].head(10).to_numpy()


# Function to evaluate the model
def test_model(coa_data, pca, pca_features, size, data_path='../data/coa_renamed/',
               test_data_path="../data/test_data.csv", test_data_secondary_path="../data/test_data_secondary.csv"):

    # Get test data
    test_data = list(csv.reader(open(test_data_path)))
    test_data_secondary = list(csv.reader(open(test_data_secondary_path)))

    # Set Score vars
    self = 0
    score = 0
    score_secondary = 0

    # Iterate over test_data
    for idx, test in enumerate(test_data):
        image_list = compare(data_path + test[0], coa_data, pca, pca_features, size)
        self += 1 if image_list[0][0] == test[0] else 0

        for idy, img in enumerate(image_list):
            # the first row is the img name itself
            if idy > 0:
                score += 1 if img[0] in test else 0
                score_secondary += 1 if img[0] in test_data_secondary[idx] else 0

    return f"score: {score}, secondary score: {score_secondary}, self: {self}/{len(test_data)}"



if __name__ == "__main__":
    # Load data
    coa_data = pd.read_csv('../data/training-data_100x100_rgb.csv')

    # Train model
    pca, pca_features, size = train_pca_model(coa_data)

    # Compare image
    compare('../data/test_data/1.png', coa_data, pca, pca_features, size)