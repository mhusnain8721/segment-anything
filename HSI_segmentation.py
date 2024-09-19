import scipy.io as sio
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.colors as mcolors

def load_mat_file(file_path, key):
    """
    Loads a .mat file and extracts the image stored under the specified key.
    Args:
        file_path (str): Path to the .mat file.
        key (str): Key under which the image data is stored.
    Returns:
        numpy.ndarray: Extracted image.
    """
    mat_data = sio.loadmat(file_path)
    if key in mat_data:
        return mat_data[key]
    else:
        raise KeyError(f"Key '{key}' not found in the .mat file.")

# Step 1: Load the RGB image from .mat file
hsi_image_path = 'image_data.mat'   # Replace with your file path
hsi_key = 'rgb_image'     # Replace if different
hsi_image = load_mat_file(hsi_image_path, hsi_key)  # Shape: (H, W, C)

print("HSI image shape:", hsi_image.shape)
print("HSI image dtype:", hsi_image.dtype)
print("HSI image min, max:", hsi_image.min(), hsi_image.max())

# Ensure the image is in uint8 format with pixel values in [0, 255]
if hsi_image.dtype != np.uint8 or hsi_image.max() <= 1.0:
    hsi_image = (hsi_image * 255).astype(np.uint8)

# Step 2: Initialize the SAM model
sam_checkpoint = 'sam_vit_h_4b8939.pth'  # Path to your SAM model checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the SAM model
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device)

# Create the SAM automatic mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Step 3: Generate segmentation predictions
masks = mask_generator.generate(hsi_image)

# Create a combined segmentation mask
height, width = hsi_image.shape[:2]
segmentation_mask = np.zeros((height, width), dtype=np.uint8)

# Assign unique labels to each mask
for idx, mask in enumerate(masks):
    segmentation_mask[mask['segmentation']] = idx + 1  # Labels start from 1

# Step 4: Visualize the segmentation result
# Create a random color map for the masks
cmap = mcolors.ListedColormap(np.random.rand(len(masks) + 1, 3))

# Plot the original image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(hsi_image)
plt.title('Original Image')
plt.axis('off')

# Plot the segmentation output
plt.subplot(1, 2, 2)
plt.imshow(hsi_image)
plt.imshow(segmentation_mask, alpha=0.5, cmap=cmap)
plt.title('Segmentation Output')
plt.axis('off')

plt.show()