import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import Adam
import os

from cyclegan_networks import UNet2Conv, UNet2ConvRes, PatchGANDiscriminator, PatchGANDiscriminator32, MBDiscriminator
from image_simulation import (generate_2d_gaussian_image4, generate_noisy_image,
                              draw_bounding_boxes,
                              generate_2d_gaussian_image_with_noise_and_boxes,
                              apply_random_mask)


def draw_bounding_boxes_on_subplots(image1, image2, boxes, i):
    """
    Draw the same rectangular bounding boxes on two different images displayed side by side.

    :param image1: 2D numpy array representing the first image.
    :param image2: 2D numpy array representing the second image.
    :param boxes: List of bounding boxes, each defined as [xmin, ymin, xmax, ymax].
    """

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Display each image
    axs[0, 0].imshow(image1, cmap='gray', origin='lower')
    axs[0, 1].imshow(image2, cmap='gray', origin='lower')
    axs[1, 0].imshow(image1, cmap='gray', origin='lower')
    axs[1, 1].imshow(image2, cmap='gray', origin='lower')

    transformed_boxes = []
    scale = (128 - 1) / 2.0  # Scaling factor from -1 to 1 range to 0 to 127 range

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        # Transform each coordinate
        xmin_transformed = (xmin + 1) * scale
        ymin_transformed = (ymin + 1) * scale
        xmax_transformed = (xmax + 1) * scale
        ymax_transformed = (ymax + 1) * scale
        transformed_boxes.append([xmin_transformed, ymin_transformed, xmax_transformed, ymax_transformed])

    # Draw each bounding box on both images
    for ax in (axs[1, 0], axs[1, 1]):
        for box in transformed_boxes:
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            rect = plt.Rectangle((xmin, ymin), width, height, linewidth=1.5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    # Save figure
    fig.savefig(f"test_folder/{i:03d}.png")
    plt.close(fig)


"""SAVE FOLDER"""

save_folder = 'test_folder/'

"""LOAD IN THE NP ARRAY WITH IMAGES"""

real_images_np = np.load('image_slices_all_augmented_ekaterina_June_2020.npy')
image_masks = np.load('image_slices_mask.npy')
print(real_images_np.shape)

""" GENERATE THE IMAGES """

print("Generate images")

# Example of generating a dataset
size = 128
num_images = 150  # 2000  # number of images in dataset

clean_images = []
bounding_boxes = []
for _ in range(num_images):
    clean_image, boxes = generate_2d_gaussian_image_with_noise_and_boxes(size, norm2_1_1=True)
    clean_images.append(clean_image)
    bounding_boxes.append(boxes)

clean_images = np.array(clean_images)
bounding_boxes = np.array(bounding_boxes)
clean_images = apply_random_mask(clean_images, image_masks)


""" PREPARE THE DATASET """

print("Prepare dataset")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Convert datasets to PyTorch tensors

clean_images_tensor = torch.tensor(clean_images).unsqueeze(1).float().to(device)

# model = UNet2ConvRes(n_channels=1, n_classes=1, num_residual_blocks=2)
model = (torch.load('training_real008_14_3_23_34_really_good/G_clean_to_noisy_150.pth'))
model.to(device)

real_fake_images = model(clean_images_tensor)

real_fake_images_np = real_fake_images.cpu().squeeze().detach().numpy()

for i in range(len(clean_images)):
    print(i)
    draw_bounding_boxes_on_subplots(clean_images[i], real_fake_images_np[i], bounding_boxes[i], i)




