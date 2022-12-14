# Helper functions
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import kornia

# Function to plot results
def plot_similar_images_grid(query, img_list, title='', sim_path='../data/coa_renamed/', img_size=128):
    # Size of the img based on the number of given images
    padding = int(img_size / 8)
    img_size_padded = int(img_size + 2 * padding)
    grid_size = int(np.ceil(len(img_list) / 5))
    grid_height = int((img_size_padded + padding) * grid_size + padding * 6) if grid_size > 1 else int(img_size_padded * 2 + padding * 6)
    grid_width = 7*img_size_padded + 2*padding

    # Create the background
    grid = Image.new('RGB',  (grid_width, grid_height), (255, 255, 255))

    # Prepare to draw text
    draw = ImageDraw.Draw(grid)
    def font(size):
        return ImageFont.truetype("Arial.ttf", size=int(size))

    # Add query image
    query_img_size = 2*img_size + 3*padding
    querry_img = Image.open(query)
    querry_img.thumbnail((query_img_size, query_img_size))
    grid.paste(querry_img, (padding, 5*padding))

    # Add titles
    draw.text((padding, padding), title, font=font(padding * 1.6), fill = (0, 0, 0))
    draw.text((padding, padding * 3), 'Query', font=font(padding*1.4), fill=(0, 0, 0))
    draw.text((query_img_size + 4 * padding, padding * 3), "Similar images", font=font(padding * 1.4), fill=(0, 0, 0))

    # Create list of positions
    positions = []
    for y in range(0, grid_size):
        for x in range(0, 5):
            pos_x = int(query_img_size + 4 * padding + x * img_size_padded)
            pox_y = int(5*padding + y * (img_size_padded + padding))
            positions.append((pos_x, pox_y))

    # Add the similar images to grid
    for idx, sim_img in enumerate(img_list):
        img = Image.open(sim_path+sim_img[0])
        img.thumbnail((img_size, img_size))
        grid.paste(img, (positions[idx][0], positions[idx][1]))

        # Add title
        text_pos_y = int(positions[idx][1] + img_size + padding*0.6)
        img_title = sim_img[0] if len(sim_img[0]) < 20 else sim_img[0][:20]+"..."
        draw.text((positions[idx][0], text_pos_y), img_title, font=font(padding * 0.8), fill=(0, 0, 0))
        draw.text((positions[idx][0], text_pos_y + padding), str(round(sim_img[1],6)), font=font(padding*0.8), fill=(50, 50, 50))

    return grid

# Function to plot multiple plots in one
def plot_plots(img_list, title=''):
    # Size of the img based on the number of given images
    width = img_list[0].width 
    height = img_list[0].height
    
    grid_size = int(np.ceil(len(img_list) / 2))
    grid_height = int(height * grid_size)
    grid_width = int(width*2)

    # Create the background
    grid = Image.new('RGB',  (grid_width, grid_height), (255, 255, 255))

    # Create list of positions
    positions = []
    for y in range(0, grid_size):
        for x in range(0, 2):
            pos_x = int(x * width)
            pox_y = int(y * height)
            positions.append((pos_x, pox_y))

    # Add the similar images to grid
    for idx, sim_img in enumerate(img_list):
        img = img_list[idx]
        img.thumbnail((width, height))
        grid.paste(img, (positions[idx][0], positions[idx][1]))

    return grid

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
def process_img(img, size, color=True, flatten=True, edge=False):
    # Resize
    resized = resize(img, size)

    if color and flatten:
        processed = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    elif color:
        processed = resized
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

        return processed

    elif flatten:
        return processed.flatten()

    else:
        return processed
