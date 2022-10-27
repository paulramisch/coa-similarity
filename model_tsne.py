# TSNE-Model zum Vergleich Suche ähnlicher Wappen in einem Datenset
# Nachteil TSNE: Es ist nicht möglich nachträglich zu transformieren, es gibt keine Mapping-Funktion.
# Neue Daten erfordern ein erneutes Training
# Außer bei Nutzung von openTSNE: https://opentsne.readthedocs.io/en/latest/index.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

# Load data
coa_data = pd.read_csv('data/training-data_40x40_rgb.csv')
coa_data.head()

# Img size: Square root of the number of columns, minus the label column (RGB)
# If greyscale: size = int((coa_data.shape[1] - 1) ** (1/2))
size = int(((coa_data.shape[1] - 1) / 3) ** (1/2))

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

    # Add cos distance to data
    coa_data_sorted = coa_data
    coa_data_sorted['cos_distance'] = similar_idx
    coa_data_sorted = coa_data_sorted.sort_values('cos_distance')

    # Plot most similar images
    plot_most_similar(coa_data_sorted.drop(['0', 'cos_distance'], axis=1))

    return coa_data_sorted[['0', 'cos_distance']].head(10).to_numpy()

# compare(coa_data[coa_data['0'] == '-1_G A lion cr..jpg'].index.values[0])
# compare(coa_data[coa_data['0'] == '-1_G E chevron.jpg'].index.values[0])
# compare(coa_data[coa_data['0'] == '-1_O B lion rampant.jpg'].index.values[0])
compare(0)