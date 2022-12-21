# Rotate files based on annoations in CSV file
# Annotations made with ImageJ
# 1. Load images as image sequence: Import > Image sequence, in dialog check option "use virtual stack"
# 2. Use angle tool: right click, to start, left click to end measure tool; new right click to start new
# Note: From left to right, angles differ otherwise
# 3. Record measurement: By pushing the key m, the measure will be recorded, in the dialog the csv can be exported
# 4. Next image can measured by using the mouse scroll wheel

import csv
import os
from PIL import Image
import math

# Parameter
img_path = "../data/coa_rotated/"
export_path = "../data/coa_rotate_export/"
angle_dict_path = "coa_rotation_angle_rounded-dict.csv"


# Function to rotate images
def rotate_img(img_path, angle):
    image = Image.open(img_path)
    # Rotation
    image_rotated = image.rotate(-angle, expand=True)

    # Transform
    # Calculate opposite cathetus from the angle and img lenhth
    a = math.tan(abs(angle) * math.pi / 180) * image.width * 0.15

    # Crop
    left = (image_rotated.width - image.width) * 0.5
    upper = (image_rotated.height - image.height) * 0.5 + a
    right = (image_rotated.width - image.width) * 0.5 + image.width
    lower = (image_rotated.height - image.height) * 0.5 + image.height + a
    image_cropped = image_rotated.crop((left, upper, right, lower))

    # image_cropped.show()

    return image_cropped


# Load dictionary of angles
with open(angle_dict_path, mode='r') as infile:
    reader = csv.reader(infile)
    angle_dict = dict((rows[0], rows[1]) for rows in reader)

# Iterate through data
for file in os.listdir(img_path):
    if file.lower().endswith(('.png', 'jpg')):
        if angle_dict.get(file) is not None:
            rotated_img = rotate_img(img_path + file, int(angle_dict.get(file)))
            rotated_img.save(export_path + file)
