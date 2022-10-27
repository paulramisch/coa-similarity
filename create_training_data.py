# Skript zur Erstellung von Trainingsdaten

import cv2
import pandas as pd
import os
import numpy as np
from helper_functions import process_img
from joblib import Parallel, delayed
from tqdm import tqdm

# Variables
number_or_parallel_jobs = 6
img_import = 'data/coa'
size = 40
rgb = True
export = 'data/training-data_{}x{}_{}.csv'.format(size, size, ("rgb" if rgb else "grey"))

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
                processed = process_img(img, size, color=rgb)

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
