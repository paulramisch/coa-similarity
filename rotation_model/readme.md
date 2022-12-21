# CNN to rotate scans of Coat of Arms
Architecture based on this blog post, which was originally written with Keras: 
https://shiva-verma.medium.com/image-angle-detection-using-neural-networks-77f38524951c

# Annotation
The foundation are 3.000 Coa of Arms, 750 of which are angled. For those 750 manual annotations were made.
The software used for the angle annotation was ImageJ: https://imagej.nih.gov/ij/

# Usage
Import the model code:
`import rotation_model as model`
`import helper_functions`

Load the trained model:
```
model = model.Model(edge_layer)
model.load_state_dict(torch.load('../data/rotation_model/rotation_model.pt'))
model.eval()
```

Predict the angle:
`helper_functions.predict_angle(model, "path/to/image.jpg", "image.jpg", 128, True)`


# Results
tested on all 3.000 images, epochs = 5 batch_size = 32:
0.9155023286759814 % within correct 5°, 0.9517631403858948 % within correct 10°,

tested on all 3.000 images, epochs = 10 (used Nr. 5) batch_size = 32:
0.9118429807052562 % within correct 5°, 0.9713905522288756 % within correct 10°

tested on all 3.000 images, epochs = 10 (used Nr. 8) batch_size = 16:
0.8938789088489687 % within correct 5°, 0.9530938123752495 % within correct 10°

tested on all 3.000 images, epochs = 10 (used Nr. 3) batch_size = 32, edge_finder_feature:
0.9051896207584831 % within correct 5°, 0.9510978043912176 % within correct 10°

tested on all 3.000 images, epochs = 10 batch_size = 32:
0.9048569527611444 % within correct 5°, 0.9540918163672655 % within correct 10°, 96 % within correct 15 %

tested on all 3.000 images, epochs = 10 batch_size = 32:,  edge_finder_feature:
0.917831004657352 % within correct 5°, 0.9750499001996008 % within correct 10°, 0.9866932801064537 % within correct 15°

tested on 3.000 images + mirrored, epochs = 10 batch_size = 32:, 9th used,  edge_finder_feature:
0.9151696606786427 % within correct 5°, 0.9770459081836327 % within correct 10°, 0.9893546240851631 % within correct 15°
