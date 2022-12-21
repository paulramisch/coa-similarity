# Skript zur Erstellung von Trainingsdaten

import cv2
import pandas as pd
import os
import numpy as np
import kornia
from joblib import Parallel, delayed
from tqdm import tqdm

# Variables
number_or_parallel_jobs = 6
img_import = 'coa_renamed'
size = 100
rgb = True
edge = True
export = 'coa_csv/training-data{}_{}x{}_{}.csv'.format(("_edge" if edge else ""), size, size, ("rgb" if rgb else "grey"))

# resize function to have the widest side being the var size and putting it on white bg
def resize(img, size=100, bg_color=0):
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
    result = np.full((size, size, 3), bg_color, dtype=np.uint8)
    result[yoff:yoff + resized.shape[0], xoff:xoff + resized.shape[1]] = resized

    return result

# Function to process image
def process_img(img, size, color=True, edge=False):
    # Resize
    resized = resize(img, size)

    # Color/Gray change
    if color:
        processed = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    else:
        # Convert to BW
        processed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # edge detection
    if edge:
        canny = kornia.filters.Canny()

        # Make image a tensor
        data = kornia.image_to_tensor(processed, keepdim=False)
        edge_layer_tensor = canny(data.float())[1]
        edge_layer = kornia.tensor_to_image(edge_layer_tensor.byte()).flatten()

        # Thumb the images
        processed = cv2.resize(processed, (int(size / 5), int(size / 5)), interpolation=cv2.INTER_AREA).flatten()

        # Combine layers
        processed = np.concatenate((processed, edge_layer), axis=0)

    else:
        # Create image vectors and return
        processed = processed.flatten()

    return processed

# Function to iterate over images
def transform_data(data):

    # Create dataframe for image vectors
    transformed_list = []

    for file in data:

        # Check if file exists
        if file.lower().endswith(('.png', 'jpg')):
            # Load image
            filepath = os.path.join(img_import, file)
            img = cv2.imread(filepath)

            if img is not None:
                processed = process_img(img, size, color=rgb, edge=edge)

                vec_df = pd.DataFrame([np.concatenate(([file], processed))])
                transformed_list.append(vec_df)
            else:
                print('NoneType Error for', file)

    transformed_data = pd.concat(transformed_list, axis=0)
    return transformed_data


# Split data in chunks of 30 images
data_blocks = np.array_split(os.listdir(img_import), (len(os.listdir(img_import))/30))

# Parallelize to process the images
count = 0
data_output = Parallel(n_jobs=number_or_parallel_jobs)(delayed(transform_data)(data) for data in tqdm(data_blocks))
print('Parallel finished')

# Combine data
training_data = pd.concat(data_output, ignore_index=True)

# Export Dataframe
training_data.to_csv(export, index=False)
print('Process finished: File saved')
