# TSNE-Model zum Vergleich Suche ähnlicher Wappen in einem Datenset
# Nachteil TSNE: Es ist nicht möglich nachträglich zu transformieren, es gibt keine Mapping-Funktion.
# Neue Daten erfordern ein erneutes Training
# Außer bei Nutzung von openTSNE: https://opentsne.readthedocs.io/en/latest/index.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import distance
from helper_functions import plot_similar_images_grid
import warnings
warnings.filterwarnings('ignore')

def train_tsne_model(coa_data, n_components=3, pca=False, pca_comp=138, learning_rate=150, perplexity=30, angle=0.2, ):
    if pca:
        # PCA anwenden
        pca, pca_features, size = train_pca_model(coa_data, False, pca_comp)

        # Output transformieren für Input in TSNE
        df_pca_features = pd.DataFrame(pca_features, columns = range(1, pca_features.shape[1] + 1))
        coa_data = coa_data[['0']].join(df_pca_features)
    
    # Img size: Square root of the number of columns, minus the label column (RGB)
    # If greyscale: size = int((coa_data.shape[1] - 1) ** (1/2))
    size = int(((coa_data.shape[1] - 1) / 3) ** (1/2))

    # Data prep
    coa_data_img = coa_data.copy().drop('0', axis=1)
    coa_data_labels = coa_data[['0']].copy()

    # TSNE
    # https://github.com/ml4a/ml4a/blob/master/examples/info_retrieval/image-tsne.ipynb
    # Max 3 dimensions
    tsne = TSNE(n_components=n_components, learning_rate=learning_rate, perplexity=perplexity, angle=angle, verbose=2)
    tsne_result = tsne.fit_transform(coa_data_img)

    return tsne_result, size


# Easy Function to plot most similiar images
def plot_most_similar(pixels, size):
    fig, axes = plt.subplots(1, 5, figsize=(6, 2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(size, size, 3))
    plt.show()


# Function to plot the grid view
def plot_similar_tsne(query, tsne_result, coa_data, path):
    query_id = coa_data[coa_data['0'] == query].index.values[0]
    image_list = compare_tsne(query_id, tsne_result, coa_data)
    return plot_similar_images_grid(path + query, image_list, 'TSNE-Model')


# Function to compare
def compare_tsne(img, tsne_result, coa_data, num=10):
    # Cosine distance comparison
    similar_idx = [ distance.cosine(tsne_result[img], feat) for feat in tsne_result ]

    # Add cos distance to data
    coa_data_sorted = coa_data[['0']].copy()
    coa_data_sorted['cos_distance'] = similar_idx
    coa_data_sorted = coa_data_sorted.sort_values('cos_distance')

    # Plot most similar images
    # plot_most_similar(coa_data_sorted.drop(['0', 'cos_distance'], axis=1), size)

    return coa_data_sorted[['0', 'cos_distance']].head(num).to_numpy()


if __name__ == "__main__":
    # Load data
    coa_data = pd.read_csv('data/training-data_100x100_rgb.csv')

    tsne_result, size = train_tsne_model(coa_data)
    # compare_tsne(coa_data[coa_data['0'] == '-1_G A lion cr..jpg'].index.values[0])
    # compare_tsne(coa_data[coa_data['0'] == '-1_G E chevron.jpg'].index.values[0])
    # compare_tsne(coa_data[coa_data['0'] == '-1_O B lion rampant.jpg'].index.values[0])
    compare_tsne(0, tsne_result, coa_data)