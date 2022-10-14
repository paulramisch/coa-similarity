# PCA-Model zum Vergleich Suche ähnlicher Wappen in einem Datenset
# Implementierung angelehnt an https://medium.com/analytics-vidhya/pca-vs-t-sne-17bcd882bf3d

# Schritte
# 1. 


# Bibliotheken laden
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Daten transformieren
# Dataframe anlegen

# PCA, Zahl der Dimensionen
pca_components = 50

# Über Bilder iterieren
for img in folder:
    img = cv2.cvtColor(cv2.imread(img)), cv2.COLOR_BGR2RGB)

    # img.shape # Gibt die Maße zurück
    # plt.imshow(img)

    # Split the image in rgb values
    r, g, b = cv2.split(img)
    r, g, b = r / 255, g / 225, b / 255

    # Show dimension
    # plt.imshow(r)

    # Zu Dataframe hinzufügen

# PCA erstellen
pca_r = PCA(n_components = pca_components)
reduced_r = pca_r.fit_transform(r)

pca_g = PCA(n_components = pca_components)
reduced_g = pca_r.fit_transform(g)

pca_b = PCA(n_components = pca_components)
reduced_b = pca_r.fit_transform(b)

# Kombinieren
combinded = np.array([reduced_r, reduced_g, reduced_b])

# Rekonstruieren
reconstruced_r = pca_r.inberse_transform(reduced_r)
reconstruced_g = pca_g.inberse_transform(reduced_g)
reconstruced_b = pca_b.inberse_transform(reduced_b)

img_reconstructed = (cv2.merge((reconstruced_r,reconstruced_g,reconstruced_b)))