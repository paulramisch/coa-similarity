# Coat of Arms similarity
Evaluation of different ML techniques to measure the similarity of Coat of Arms

## Methoden
- PCA: Principal Component Analysis 
Unsupervised Learning Technik um Dimensionalität von Daten zu reduzieren, mithilfe von orthogonal linear transformation (Kumar 2019)
Eher globale Ähnlichkeit
- t-SNE 
Wie PCA zur Reduktion der Dimensionalität von Daten, aber mit non-linear dimensionality reduction; zudem iterativ (Kumar 2019)
Vorteil offenbar: Fokus auf lokale Ähnlichkeiten
- UMAP
Ebenfalls Technik zur Dimenionsionalitätsreduktion, aber 

Bei PCA zu beachten: Bildausschnitte müssen gut gewählt sein; also Wappen mitting & ähnlich viel Hintergrund; gleiche Auflösung

### Principal Component Analysis
1. Umwandeln der vorhandenen Wappen in Dataframe
Ziel nach Umwandeln: Erste Spalte: Label (Dateiname), die weiteren n (length x height Pixel) Spalten
Jede Zeile repräsentiert ein Bild; Umwandeln in Grautöne


### Ressourcen
#### Allgemein

Erdogan Taskesen, Quantitative comparisons between t-SNE, UMAP, PCA, and Other Mappings, medium.com Towards Data Science, 2022,
https://towardsdatascience.com/the-similarity-between-t-sne-umap-pca-and-other-mappings-c6453b80f303 (abgerufen: 12.10.22).

#### PCA:
Gute Ressourcen für PCA wurden für Eigenfaces geschrieben, wahrscheinlich möglich, solches nachzuahmen:

Implementierung:
Dario Radečić, Eigenfaces — Face Classification in Python, medium.com Towards Data Science 2020,
https://towardsdatascience.com/eigenfaces-face-classification-in-python-7b8d2af3d3ea (abgerufen: 14.10.22).

The eigenfaces example: chaining PCA and SVMs, scipy-lectures.org/, 
https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_eigenfaces.html (abgerufen: 14.10.22).

(mit Google Colab sheet)
Wenjing Liu, How to Get Eigenfaces, medium.com 2013,
https://medium.com/@lwj.liuwenjing/how-to-get-eigenfaces-a9caeeba8767 (abgerufen: 14.10.22).

Github
https://github.com/vutsalsinghal/EigenFace/tree/master/Dataset
***

Andere:

Miller, Max, The Basics: Principal Component Analysis, medium.com Towards Data Science 2020,
https://towardsdatascience.com/the-basics-principal-component-analysis-83c270f1a73c (abgerufen: 14.10.22).

Santhosh Kumar R, Principal Component Analysis: In-depth understanding through image visualization, medium.com Towards Data Science 2019,
https://towardsdatascience.com/principal-component-analysis-in-depth-understanding-through-image-visualization-892922f77d9f (abgerufen: 12.10.22).

Manpreet Singh Minhas, Visualizing feature vectors/embeddings using t-SNE and PCA, medium.com Towards Data Science 2021,
https://towardsdatascience.com/visualizing-feature-vectors-embeddings-using-pca-and-t-sne-ef157cea3a42 (abgerufen: 12.10.22).

#### Implementierung PCA, t-SNE
Namratesh Shrivastav, PCA vs t-SNE: which one should you use for visualization, medium.com Analytics Vidhya 2019,
https://medium.com/analytics-vidhya/pca-vs-t-sne-17bcd882bf3d (abgerufen: 14.10.22)

#### UMAP:
Nikolay Oskolkov, How Exactly UMAP Works, medium.com Towards Data Science 2019,
https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668 (abgerufen: 12.10.22).

#### Auto-Encoders:
Aditya Oke, Image Similarity Search in PyTorch, medium.com PyTorch,
https://medium.com/pytorch/image-similarity-search-in-pytorch-1a744cf3469 (abgerufen: 12.10.22).

Alin Cijov, Image Similarity Search in PyTorch, kaggle,
https://www.kaggle.com/code/alincijov/image-similarity-search-in-pytorch (abgerufen: 12.10.22).

#### Pretrained model:
Sascha Heyer, How to Implement Image Similarity Using Deep Learning, medium.com Towards Data Science, 2022,
https://towardsdatascience.com/image-similarity-with-deep-learning-c17d83068f59 (abgerufen: 12.10.22).

Maciej D. Korzec, Recommending Similar Images Using PyTorch, Full transfer learning implementation with Resnet18, medium.com Towards Data Science 2020,
https://towardsdatascience.com/recommending-similar-images-using-pytorch-da019282770c (abgerufen: 12.10.22).

#### Feature based alignment (um PCA performance zu verbessern):
Satya Mallick, Feature Based Image Alignment using OpenCV (C++/Python), learnopencv.com 2018,
https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/ (abgerufen: 17.10.22).

### Abstrakte Maße
- Farb-Histogramm
- Korrelogramm
- Histogram-of-Gradients

Evtl. auch:
Autoencoder basiertes Embedding als Bildähnlichkeitsmaß verwendet wird (ähnlich zu PCA). 

### Leistung
Machine Learning und t-SNE / UMAP brauchen Rechenpower, aber managebar.
Die anderen Verfahren müssten easy auf jedem Rechner klappen.

## Evaluation
Bildung einer Stichprobe von zwanzig Wappen, zwei Evaluationen:
- Ähnlichkeit des ausgewählten Wappens mit Top 1
- Ähnlichkeit des ausgewählten Wappens mit Top 5

Grundlage für Entscheidung für systemnutzung
