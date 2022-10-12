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

### Ressourcen
Allgemein
Sascha Heyer, How to Implement Image Similarity Using Deep Learning, medium.com Towards Data Science, 2022,
https://towardsdatascience.com/image-similarity-with-deep-learning-c17d83068f59 (abgerufen: 12.10.22).

Erdogan Taskesen, Quantitative comparisons between t-SNE, UMAP, PCA, and Other Mappings, medium.com Towards Data Science, 2022,
https://towardsdatascience.com/the-similarity-between-t-sne-umap-pca-and-other-mappings-c6453b80f303 (abgerufen: 12.10.22).

PCA:
Santhosh Kumar R, Principal Component Analysis: In-depth understanding through image visualization, medium.com Towards Data Science 2019,
https://towardsdatascience.com/principal-component-analysis-in-depth-understanding-through-image-visualization-892922f77d9f (abgerufen: 12.10.22).

Manpreet Singh Minhas, Visualizing feature vectors/embeddings using t-SNE and PCA, medium.com Towards Data Science 2021,
https://towardsdatascience.com/visualizing-feature-vectors-embeddings-using-pca-and-t-sne-ef157cea3a42 (abgerufen: 12.10.22).

UMAP:
Nikolay Oskolkov, How Exactly UMAP Works, medium.com Towards Data Science 2019,
https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668 (abgerufen: 12.10.22).

Auto-Encoders:
Aditya Oke, Image Similarity Search in PyTorch, medium.com PyTorch,
https://medium.com/pytorch/image-similarity-search-in-pytorch-1a744cf3469 (abgerufen: 12.10.22).

Alin Cijov, Image Similarity Search in PyTorch, kaggle,
https://www.kaggle.com/code/alincijov/image-similarity-search-in-pytorch (abgerufen: 12.10.22).

Pretrained model:
Maciej D. Korzec, Recommending Similar Images Using PyTorch, Full transfer learning implementation with Resnet18, medium.com Towards Data Science 2020,
https://towardsdatascience.com/recommending-similar-images-using-pytorch-da019282770c  (abgerufen: 12.10.22).

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