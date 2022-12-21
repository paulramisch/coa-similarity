import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def plot_similar_images_grid(img_list, short=False, sim_path="coa_renamed/", img_size=128):
    # Size of the img based on the number of given images
    padding = int(img_size / 3)
    img_size_padded = int(img_size + 1 * padding)
    len_list = len(img_list) if len(img_list) > 0 else 1
    grid_size = int(np.ceil(len_list / 3))
    grid_height = int((img_size_padded + padding) * grid_size)
    grid_width = 7*img_size_padded if not short else grid_height

    # Create the background
    grid = Image.new('RGB',  (grid_width, grid_height), (255, 255, 255))

    # Prepare to draw text
    draw = ImageDraw.Draw(grid)
    def font(size):
        return ImageFont.truetype("Arial.ttf", size=int(size))

    # Create list of positions
    positions = []
    for y in range(0, grid_size):
        for x in range(0, 3):
            pos_x = int(padding + x * img_size_padded)
            pox_y = int(padding + y * (img_size_padded + padding))
            positions.append((pos_x, pox_y))

    # Add the similar images to grid
    for idx, sim_img in enumerate(img_list):
        try:
            img = Image.open(sim_path+sim_img)
            img.thumbnail((img_size, img_size))
        except:
            print("An exception occurred: Image not found")
            img = Image.new('RGB',  (img_size, img_size), (180, 180, 180))
            size = int(img_size/10)
            ImageDraw.Draw(img).text((size, size), 'Image Loading Error', font=ImageFont.truetype("Arial.ttf", size=size), fill=(0, 0, 0))

        grid.paste(img, (positions[idx][0], positions[idx][1]))

    return grid


def plot_plots(img_list, title=''):
    # Size of the img based on the number of given images
    height = max([img.height for img in img_list]) / 2
    width = img_list[1].width

    grid_size = int(np.ceil(len(img_list) / 3))
    grid_height = int(height * grid_size)
    grid_width = int(width * 2) + img_list[0].width

    # Create the background
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

    # Create list of positions
    positions = []
    for y in range(0, grid_size):
        for x in range(0, 3):
            if x == 0:
                positions.append((0, int(y * height)))
            else:
                pos_x = int((x - 1) * width + img_list[0].width)
                pox_y = int(y * height)
                positions.append((pos_x, pox_y))

    # Add the similar images to grid
    for idx, sim_img in enumerate(img_list):
        img = img_list[idx]
        if img.width > 200:
            img.thumbnail((width, height))
        grid.paste(img, (positions[idx][0], positions[idx][1]))

    return grid

# Set test_data path
test_data_path = "test_data.csv"
test_data_secondary_path = "test_data_secondary.csv"

# Load test data
test_data = list(csv.reader(open(test_data_path)))
test_data_secondary = list(csv.reader(open(test_data_secondary_path)))

# plot
i = 0
plots = []
for idx, img_list in enumerate(test_data):
    plots.append(plot_similar_images_grid([img_list[0]], True))

    img_list_primary = list(filter(None, img_list[1:]))
    plots.append(plot_similar_images_grid(img_list_primary))

    img_list_secondary = list(filter(None, test_data_secondary[idx][1:]))
    plots.append(plot_similar_images_grid(img_list_secondary))

plot = plot_plots(plots)
plot.save("plots/test_data.png")
