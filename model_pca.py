# PCA-Model zum Vergleich Suche ähnlicher Wappen in einem Datenset
# Implementierung angelehnt an https://medium.com/analytics-vidhya/pca-vs-t-sne-17bcd882bf3d
# https://towardsdatascience.com/eigenfaces-face-classification-in-python-7b8d2af3d3ea

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')


# Load data
coa_data = pd.read_csv('data/pca_training_data.csv')
coa_data.head()


# Img size: Square root of the number of columns, minus the label column
size = int((coa_data.shape[1] - 1) ** (1/2))


# Visualisation
def plot_coa(pixels):
    fig, axes = plt.subplots(5, 5, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(size, size), cmap='gray')
    plt.show()


plot_coa(coa_data.drop('0', axis=1))


# Data prep
X = coa_data.drop('0', axis=1)
y = coa_data['0']


# PCA
X_train, X_test, y_train, y_test = train_test_split(X, y)
pca = PCA().fit(X_train)


# Visual analysis of the number of needed variables
plt.figure(figsize=(18, 7))
plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3)
# plt.show()


# Second training with compontent number
pca = PCA(n_components=20).fit(X_train)
X_train_pca = pca.transform(X_train)


# Building comparision engine