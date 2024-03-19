import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.utils as vutils
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
import os
from datetime import datetime
import json
from PIL import Image
from torch.optim.lr_scheduler import ExponentialLR

from utensils import (FIDCalculator, adjust_learning_rate, show_images, show_images2, plot_losses, plot_D_losses,
                      random_flip_tensors, compute_gradient_penalty)
from cyclegan_networks import (UNet2Conv, UNet2ConvRes, PatchGANDiscriminator, PatchGANDiscriminator32, MBDiscriminator,
                               UNet2ConvTrans, UNet2ConvResLarge, UNet2ConvResAOT) #, UNetGeneratorWithTransformer)
from cyclegan_networks import UNetGeneratorWithTransformer

from image_simulation import (generate_2d_gaussian_image, generate_2d_gaussian_image2, generate_2d_gaussian_image3,
                              generate_2d_gaussian_image4, generate_noisy_image, generate_noisy_image2,
                              apply_random_mask,
                              generate_2d_gaussian_image_with_noise_and_boxes)


class PatchInpaintingDataset(Dataset):
    def __init__(self, images_array, patch_size=16, mask_prob=0.4, transform=None, image_height=128, image_width=128):
        """
        images_array: NumPy array of shape (num_images, height, width).
        patch_size: Size of the square patch.
        mask_prob: Probability to mask each patch.
        transform: PyTorch transforms to apply to each image.
        image_height: Height of the images.
        image_width: Width of the images.
        """
        self.images = images_array
        self.patch_size = patch_size
        self.mask_prob = mask_prob
        self.transform = transform
        self.image_height = image_height
        self.image_width = image_width

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image and convert it to a tensor, adding channel dimension if necessary
        image = self.images[idx]
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Ensure image is a tensor, adding channel dimension if it's missing
            if not torch.is_tensor(image):
                image = torch.tensor(image, dtype=torch.float32)
            if image.dim() == 2:  # If there's no channel dimension
                image = image.unsqueeze(0)

        # Check if image dimensions match the expected size, resize if necessary
        if image.size(1) != self.image_height or image.size(2) != self.image_width:
            raise ValueError(
                f"Expected image size {self.image_height}x{self.image_width}, but got {image.size(1)}x{image.size(2)}")

        # Initialize mask with ones and set masked regions to -1
        mask = torch.ones_like(image)
        for i in range(0, self.image_height, self.patch_size):
            for j in range(0, self.image_width, self.patch_size):
                if np.random.rand() < self.mask_prob:
                    mask[:, i:i + self.patch_size, j:j + self.patch_size] = -1

        # Apply the mask
        masked_image = image * (mask != -1).float() + mask * (mask == -1).float()

        return masked_image, mask, image  # Return the masked image, the mask, and the original image


def save_plot_comparison(input_images, masked_images, reconstructed_images, epoch, file_path='output', num_examples=3):
    """
    Plots and saves a comparison between batches of input, masked, and reconstructed images,
    showing 'num_examples' instances from each batch side by side.

    Args:
    - input_images (Tensor): The batch of original unmasked images.
    - masked_images (Tensor): The batch of masked images fed into the generator.
    - reconstructed_images (Tensor): The batch of output images from the generator.
    - epoch (int): Current epoch number for labeling the saved file.
    - file_path (str): Path to save the output image.
    - num_examples (int): Number of examples to plot from each batch.
    """
    # Ensure input tensors are moved to CPU and detach them from computation graph
    input_images = input_images.cpu().detach()[:num_examples]
    masked_images = masked_images.cpu().detach()[:num_examples]
    reconstructed_images = reconstructed_images.cpu().detach()[:num_examples]

    # Concatenate the images horizontally for each example
    comparison = torch.cat((input_images, masked_images, reconstructed_images), dim=0)

    # Plot and save the figure
    plt.figure(figsize=(10, 10))
    grid = vutils.make_grid(comparison, nrow=num_examples, normalize=True, scale_each=True, padding=2)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f'Epoch {epoch}: Input, Masked, and Reconstructed Images', fontsize=12)
    plt.savefig(f'{file_path}/comparison_epoch_{epoch:03d}.png')
    plt.close()


"""CREATE DIRECTORY AND LOAD TRAINING IMAGES"""
real_images_np = np.load('image_slices_all_augmented_ekaterina_June_2020.npy')  # dataset (numpy) of real images
# real_images_np = np.load('images_256_felix.npy')  # dataset (numpy) of real images
# real_images_np = np.load('whole_images_512_512.npy')
image_masks = np.load('image_slices_mask.npy')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

"""GENERATE SIMULATED IMAGES"""

print("Generate images")
num_images = real_images_np.shape[0]
size = 128
clean_images = []
for _ in range(num_images):
    clean_image, boxes = generate_2d_gaussian_image_with_noise_and_boxes(size, norm2_1_1=True)
    # clean_image = generate_2d_gaussian_image4(size, norm2_1_1=False)  # create the images
    # clean_images_min_noise = generate_noisy_image(clean_image, noise_level_low=0.03, noise_level_high=0.03,
    #                                               norm2_1_1=True)  # add slight noise to promote backpropagation
    # clean_images.append(clean_images_min_noise)
    clean_images.append(clean_image)
clean_images = np.array(clean_images)
clean_images = apply_random_mask(clean_images, image_masks)

"""DATASETS AND DATALOADERS"""

print("Prepare dataset")

# Convert datasets to PyTorch tensors
# clean_images_tensor = torch.tensor(clean_images).unsqueeze(1).float()
# noisy_images_tensor = torch.tensor(real_images_np).unsqueeze(1).float()

data_combined = np.concatenate([real_images_np, clean_images], axis=0)

# dataset = PatchInpaintingDataset(real_images_np, transform=None, patch_size=32, image_width=256, image_height=256)
dataset = PatchInpaintingDataset(data_combined, transform=None, patch_size=16, image_width=128, image_height=128)

# dataset = PatchInpaintingDataset(real_images_np, transform=None, patch_size=16, image_width=512, image_height=512)
dataloader = DataLoader(dataset, batch_size=25, shuffle=True)


""" Initialize the generator and discriminator """

print("Initialize the generator and discriminator")

"""NEURAL NETS"""

# G_clean_to_noisy = UNet2ConvTrans(n_channels=1, n_classes=1, num_transformer_blocks=12, depth=2, heads=8, mlp_dim=2048)
# G_clean_to_noisy = UNet2ConvRes(n_channels=1, n_classes=1, num_residual_blocks=10)
# G_clean_to_noisy = UNet2ConvResLarge(n_channels=1, n_classes=1, num_residual_blocks=10)
G_clean_to_noisy = UNetGeneratorWithTransformer()

G_clean_to_noisy.to(device)
# G_noisy_to_clean.to(device)

"""OPTIMIZERS"""

# Define loss function and optimizer
criterion = nn.MSELoss()

llr = 0.0001
optimizer = Adam(G_clean_to_noisy.parameters(), lr=llr)  # lr=0.001 0.0001 0.00001
# scheduler = ExponentialLR(optimizer, gamma=0.99)

num_epochs = 1000

for epoch in range(num_epochs):
    for i, (masked_images, masks, original_images) in enumerate(dataloader):
        masked_images = masked_images.to(device)
        original_images = original_images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = G_clean_to_noisy(masked_images)

        # Calculate the loss ONLY within the masked region
        # The model should only be penalized for inaccuracies within the masked area
        loss = criterion(outputs * masks, original_images * masks)
        # loss = criterion(outputs, original_images)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        if i == 0:  # Just take the first batch
            # Assume the generator outputs for the masked images
            reconstructed_images = G_clean_to_noisy(masked_images.to(device)).to('cpu')

            # Call the updated function to plot and save three examples
            save_plot_comparison(original_images, masked_images, reconstructed_images, epoch,
                                 file_path='output_aot',
                                 num_examples=3)
        if (epoch % 10 == 0) and (epoch != 0) and (i == 0):
            model_path = "output_resnet10/G_pretrained_evit.pth"
            torch.save(G_clean_to_noisy.state_dict(), model_path)

    # scheduler.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
