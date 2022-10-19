# Skript zur Erstellung von Trainingsdaten

import cv2
import pandas as pd
import os
import numpy as np

# Variables
img_import = 'data/coa'
export = 'data/pca_training_data.csv'
size = 100

# Create dataframe for image vectors
training_data = pd.DataFrame()


# resize function to have the widest side being the var size
def resize(img):
    # Set resize factors
    f1 = size / img.shape[0]
    f2 = size / img.shape[1]

    # Get smaller factor and compute dimensions
    f = min(f1, f2)
    dim = (int(img.shape[1] * f), int(img.shape[0] * f))

    # Resize
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Pad with white background
    # Compute xoff and yoff for padding
    yoff = round((size - resized.shape[0]) / 2)
    xoff = round((size - resized.shape[1]) / 2)

    # Combine the two
    result = np.full((size, size, 3), 255, dtype = np.uint8)
    result[yoff:yoff + resized.shape[0], xoff:xoff + resized.shape[1]] = resized

    return result


# Function to process image
def process_img(img):
    # Resize
    resized = resize(img)

    # Convert to BW
    processed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Create image vectors and return
    return processed.flatten()


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
            processed = process_img(img)

            vec_df = pd.DataFrame([np.concatenate(([file], processed))])
            training_data = pd.concat([training_data, vec_df], axis=0)
        else:
            print('NoneType Error for', file)

# Export Dataframe
training_data.to_csv(export, index=False)
print('Process finished')