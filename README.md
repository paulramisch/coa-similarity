# Coat of Arms similarity
Evaluation of different ML techniques to measure the similarity of Coat of Arms

## Methoden
- PCA: könnte schon ganz gut klappen 
(falls Bildausschnitte gut gewählt sind; also Wappen mitting & ähnlich viel Hintergrund; gleiche Auflösung)

Ebenfalls testen:
- t-SNE 
- UMAP 

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