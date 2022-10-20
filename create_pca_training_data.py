# Skript zur Erstellung von Trainingsdaten

import cv2
import pandas as pd
import os
import numpy as np
from helper_functions import process_img_pca

# Variables
img_import = 'data/coa'
export = 'data/pca_training_data.csv'
size = 100

# Create dataframe for image vectors
training_data = pd.DataFrame()

# Iterate over pictures
count = 0
len_files = len(os.listdir(img_import))
for file in os.listdir(img_import):
    count += 1

    # Check if file exists
    if file.lower().endswith(('.png', 'jpg')):
        # Load image
        filepath = os.path.join(img_import, file)
        img = cv2.imread(filepath)

        if img is not None:
            print(round(count / len_files * 100), '% - Processing', file)
            processed = process_img_pca(img, 100)

            vec_df = pd.DataFrame([np.concatenate(([file], processed))])
            training_data = pd.concat([training_data, vec_df], axis=0)
        else:
            print('NoneType Error for', file)

# Export Dataframe
training_data.to_csv(export, index=False)
print('Process finished')