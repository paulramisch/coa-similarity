# CNN tom rotate scans of Coat of Arms
Architecture based on this architecture which was originally written with Keras: 
https://shiva-verma.medium.com/image-angle-detection-using-neural-networks-77f38524951c

# Annotation
The foundation are 3.000 Coa of Arms, 750 of which are angled. For those 750 manual annotations were made.
The software used for the angle annotation was ImageJ: https://imagej.nih.gov/ij/

# Results
tested on all 3.000 images, epochs = 5 batch_size = 32:
0.9155023286759814 % within right 5%, 0.9517631403858948 % within right 10%,

tested on all 3.000 images, epochs = 10 (used Nr. 5) batch_size = 32:
0.9118429807052562 % within right 5%, 0.9713905522288756 % within right 10%

tested on all 3.000 images, epochs = 10 (used Nr. 8) batch_size = 16:
0.8938789088489687 % within right 5%, 0.9530938123752495 % within right 10%

tested on all 3.000 images, epochs = 10 (used Nr. 3) batch_size = 32, edge_finder_feature:
0.9051896207584831 % within right 5%, 0.9510978043912176 % within right 10%

tested on all 3.000 images, epochs = 10 batch_size = 32:
0.9048569527611444 % within right 5%, 0.9540918163672655 % within right 10%, 96 % within right 15 %