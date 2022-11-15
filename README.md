# Coat of Arms similarity
Evaluation of different ML techniques to measure the similarity of Coat of Arms

PCA
TSNE
UMAP
CNN: Welche Funktion haben die verschiedenen Layer

Evtl. bei Sebastian Kiel nachfragen, Retina-Net, CNN für Drehungen, evtl. verschiedene geometrische Strukturen
evtl. für Drehung

Aktuell annotierter Stand, Farbbias
Styletransfer zu CNN um Trainingsdaten zu generieren; 15.000, ggf. tilten
Frage an Sebastian, wie weit ist die Richtung wichtig

Trainingsdatenderivate generieren 

Problem: Ähnlichkeit zwischen gedrehten Bild
Modell Training mit CNN

Welche Architektur fürs Training? Yolo 4


Decoder:
- Bild
+ Dateinummer


# Todos
[] Erstellung eines notebooks
[] 

## Methoden
- PCA: Principal Component Analysis 
Unsupervised Learning Technik um Dimensionalität von Daten zu reduzieren, mithilfe von orthogonal linear transformation (Kumar 2019)
Eher globale Ähnlichkeit
- t-SNE 
Wie PCA zur Reduktion der Dimensionalität von Daten, aber mit non-linear dimensionality reduction; zudem iterativ (Kumar 2019)
Vorteil offenbar: Fokus auf lokale Ähnlichkeiten
- UMAP
Ebenfalls Technik zur Dimenionsionalitätsreduktion, aber 
- CNN


Dinge, dich noch zu tun sind: https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/
Bei PCA zu beachten: Bildausschnitte müssen gut gewählt sein; also Wappen mitting & ähnlich viel Hintergrund; gleiche Auflösung

### Principal Component Analysis
1. Umwandeln der vorhandenen Wappen in Dataframe
Ziel nach Umwandeln: Erste Spalte: Label (Dateiname), die weiteren n (length x height Pixel) Spalten
Jede Zeile repräsentiert ein Bild; Umwandeln in Grautöne


### Ressourcen
#### Allgemein
Erdogan Taskesen, Quantitative comparisons between t-SNE, UMAP, PCA, and Other Mappings, medium.com Towards Data Science, 2022,
https://towardsdatascience.com/the-similarity-between-t-sne-umap-pca-and-other-mappings-c6453b80f303 (abgerufen: 12.10.22).

Anson Wong, Image Retrieval (via Autoencoders / Transfer Learning), Github 2019,
https://github.com/ankonzoid/artificio/tree/master/image_retrieval (abgerufen: 25.10.22).

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
Phillip Lippe, Tutorial 9: Deep Autoencoders, UVA readthedocs.ui 2022,
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html (abgerufen: 25.10.22).

Mitch Jablonski, Convolutional Autoencoder, udacity Github 2021,
https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/convolutional-autoencoder/Convolutional_Autoencoder_Solution.ipynb (abgerufen: 25.10.22).

Aditya Oke, Image Similarity Search in PyTorch, medium.com PyTorch,
https://medium.com/pytorch/image-similarity-search-in-pytorch-1a744cf3469 (abgerufen: 12.10.22).

Alin Cijov, Image Similarity Search in PyTorch, kaggle,
https://www.kaggle.com/code/alincijov/image-similarity-search-in-pytorch (abgerufen: 12.10.22).

#### Pretrained model - Transfer Learning
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

https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1
https://medium.com/@gabriel_83172/how-to-find-similar-images-using-math-a-gentile-introduction-to-image-retrieval-and-neural-67f3c987b643

_transformed score: 11, secondary score: 5
_stn: 7, secondary: 4
_transformed2 score: 7, secondary score: 3
_transformed2 bw: score: 8, secondary score: 4
_transformed3 score: 11, secondary score: 6

Epochs = 4, Training Loss : 0.03057108446955681
Epochs = 4, Validation Loss : 0.017935700714588165

Epochs = 9, Training Loss : 0.05654442310333252
Epochs = 9, Validation Loss : 0.02859371528029442

Epochs = 8, Training Loss : 0.0351361483335495
Epochs = 8, Validation Loss : 0.020298205316066742

Epochs = 8, Training Loss : 0.027494631707668304
Epochs = 8, Validation Loss : 0.016442641615867615

Epochs = 8, Validation Loss : 0.016699939966201782
Epochs = 9, Training Loss : 0.029162311926484108

Epochs = 7, Training Loss : 0.026608875021338463
Epochs = 7, Validation Loss : 0.015363448299467564

Epochs = 6, Training Loss : 0.04016155004501343
Epochs = 6, Validation Loss : 0.024223677814006805

Epochs = 8, Training Loss : 0.03813875466585159
Epochs = 8, Validation Loss : 0.02090904489159584

Epochs = 8, Training Loss : 0.03716358542442322
Epochs = 8, Validation Loss : 0.019994445145130157

Epochs = 9, Training Loss : 0.035220108926296234
Epochs = 9, Validation Loss : 0.018688756972551346