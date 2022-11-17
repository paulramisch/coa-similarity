import os
import torch
import rotation_model as model
import csv
import helper_functions

# Load model
model = model.Model()
model.load_state_dict(torch.load('../data/rotation_model/rotation_model.pt'))
model.eval()

# Load img dict
with open('../data/coa_rotation_angle_rounded-dict.csv', mode='r') as infile:
    reader = csv.reader(infile)
    angle_dict = dict((rows[0], rows[1]) for rows in reader)

# List of all img
img_path = '../data/coa_renamed'
all_imgs = []
for file in os.listdir(img_path):
    # check only text files
    if file.lower().endswith(('.png', 'jpg')):
        all_imgs.append(file)

# Set counter
true_5 = 0
true_10 = 0
true_15 = 0

# Check img quality
for img_title in all_imgs:
    prediction = int(helper_functions.predict_angle(model, img_path, img_title, size=128))
    real = int(angle_dict.get(img_title, 0))
    difference = real - prediction

    # print(f"{img_title}: angle: {real}, prediction: {prediction}, difference: {difference}")

    true_5 += 1 if abs(difference) < 5 else 0
    true_10 += 1 if abs(difference) < 10 else 0
    true_15 += 1 if abs(difference) < 15 else 0

rel_5 = true_5 / len(all_imgs)
rel_10 = true_10 / len(all_imgs)
rel_15 = true_15 / len(all_imgs)

print(f"{rel_5} % within right 5%, {rel_10} % within right 10%, {rel_15} % within right 15 %")
