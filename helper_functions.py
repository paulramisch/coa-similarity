# Helper functions
import cv2
import numpy as np


# resize function to have the widest side being the var size and putting it on white bg
def resize(img, size=100, bg_color=255):
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
    result = np.full((size, size, 3), bg_color, dtype = np.uint8)
    result[yoff:yoff + resized.shape[0], xoff:xoff + resized.shape[1]] = resized

    return result


# Function to process image
def process_img(img, size, color=True):
    # Resize
    resized = resize(img, size)

    if color:
        processed = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    else:
        # Convert to BW
        processed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Create image vectors and return
    return processed.flatten()