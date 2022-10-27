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
size = 96
rgb = True
export = 'data/coa_resized'


# Function to iterate over images
def transform_data(data):
    for file in data:
        # Check if file exists
        if file.lower().endswith(('.png', 'jpg')):
            # Load image
            filepath = os.path.join(img_import, file)
            img = cv2.imread(filepath)

            if img is not None:
                processed = process_img(img, size, color=True, flatten=False)
                if not cv2.imwrite(export + '/' + file, processed):
                    print('Error for', file)

            else:
                print('NoneType Error for', file)

# Split data in chunks of 30 images
data_blocks = np.array_split(os.listdir(img_import), (len(os.listdir(img_import))/30))

# Parallelize to process the images
count = 0
Parallel(n_jobs=number_or_parallel_jobs)(delayed(transform_data)(data) for data in tqdm(data_blocks))
print('Parallel finished')

