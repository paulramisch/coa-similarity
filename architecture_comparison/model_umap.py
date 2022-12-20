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
from helper_functions import process_img, plot_similar_images_grid
import warnings
import csv
warnings.filterwarnings('ignore')


def train_umap_model(coa_data, n_neighbors=100, min_dist=0.3, n_components=30, edge=False, greyscale=False):
    # Load data
    np.random.seed(42)
    coa_data.head()

    # Img size: Square root of the number of columns, minus the label column (RGB)
    # If greyscale: size = int((coa_data.shape[1] - 1) ** (1/2))
    if edge:
        size = int(((coa_data.shape[1] - 1) / 28) ** (1 / 2) * 5)
    elif greyscale:
        size = int((coa_data.shape[1] - 1) ** (1 / 2))
    else:
        # Img size: Square root of the number of columns, minus the label column (RGB)
        size = int(((coa_data.shape[1] - 1) / 3) ** (1/2))

    # Data prep
    coa_data_img = coa_data.copy().drop('0', axis=1) / 257
    coa_data_labels = coa_data[['0']].copy()

    # UMAP
    # For parameters: https://umap.scikit-tda.org/parameters.html
    trans = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components).fit(coa_data_img)

    return trans, size


# Easy Function to plot most similar images
def plot_most_similar(pixels, size):
    fig, axes = plt.subplots(1, 5, figsize=(6, 2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(size, size, 3))
    plt.show()


# Function to plot the grid view
def plot_similar_umap(query, trans, coa_data, size, edge=False):
    image_list = compare_umap(query, trans, coa_data, size, edge)
    return plot_similar_images_grid(query, image_list, 'UMAP-Model')


# Function to compare
def compare_umap(img_src, trans, coa_data, size, edge=False):
    # Load image
    img = cv2.imread(img_src)

    # Transform image
    coa = process_img(img, size, edge=edge) / 255

    # Reformat as DataFrame
    coa_df = pd.DataFrame([np.concatenate((['coa'], coa))]).drop(0, axis=1)

    # Extract features
    img_features = trans.transform(coa_df)

    # Cosine distance comparison
    similar_idx = [distance.euclidean(img_features[0], feat) for feat in trans.embedding_]

    # Add cos distance to data
    coa_data_sorted = coa_data[['0']].copy()
    coa_data_sorted['euc_distance'] = similar_idx
    coa_data_sorted = coa_data_sorted.sort_values('euc_distance')

    # Plot most similar images
    # plot_most_similar(coa_data_sorted.drop(['0', 'euc_distance'], axis=1))

    return coa_data_sorted[['0', 'euc_distance']].head(10).to_numpy()


def test_model_umap(coa_data, trans, size, edge=False, data_path='../data/coa_renamed/',
                    test_data_path="../data/test_data.csv",
                    test_data_secondary_path="../data/test_data_secondary.csv"):

    # Get test data
    test_data = list(csv.reader(open(test_data_path)))
    test_data_secondary = list(csv.reader(open(test_data_secondary_path)))

    # Set Score vars
    self = 0
    self_10 = 0
    score = 0
    score_secondary = 0

    # Iterate over test_data
    for idx, test in enumerate(test_data):
        image_list = compare_umap(data_path + test[0], trans, coa_data, size, edge)
        self += 1 if image_list[0][0] == test[0] else 0

        for idy, img in enumerate(image_list):
            score += 1 if img[0] in test[1:] else 0
            score_secondary += 1 if img[0] in test_data_secondary[idx][1:] else 0
            self_10 += 1 if img[0] == test[0] else 0

    return f"score: {score}, secondary score: {score_secondary}, self: {self}/{len(test_data)}, self in top-10: {self_10}/{len(test_data)}"


if __name__ == "__main__":
    # Load data
    coa_data = pd.read_csv('../data/training-data_100x100_rgb.csv')
    trans, size = train_umap_model(coa_data)

    compare_umap('../data/test_data/-1_G A lion cr..jpg', trans, coa_data, size)
    test_model_umap(coa_data, trans, size)

    # Test
    # Umap is stochastic: Not always the same: https://github.com/lmcinnes/umap/issues/566
    # title = '-1_G A lion cr..jpg'
    # a = trans.transform(coa_data[coa_data['0'] == title].drop(['0', 'euc_distance'], axis=1))[0]
    # b = trans.transform(coa_data.drop(['0', 'euc_distance'], axis=1))[coa_data[coa_data['0'] == title].index.values[0]]
    # print(distance.euclidean(a, b))

    # Distance: euclidean distance instead of cosine, because Umap is in euclidean space
    # https://github.com/lmcinnes/umap/issues/519

