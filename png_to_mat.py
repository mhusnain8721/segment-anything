from PIL import Image
import numpy as np
import scipy.io

# Load the PNG image
img = Image.open('rgb.png')

# Convert the image to RGB (if it's not already)
img_rgb = img.convert('RGB')

# Convert image data to a NumPy array
img_array = np.array(img_rgb)

# Prepare a dictionary to save in the .mat file
mat_data = {'rgb_image': img_array}

# Save the data to a .mat file with the key 'rgb_image'
scipy.io.savemat('image_data.mat', mat_data)

print("Image successfully converted and saved as .mat file.")