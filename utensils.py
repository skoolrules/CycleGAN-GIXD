from scipy.linalg import sqrtm
import numpy as np
import torch
from torchvision.models import inception_v3
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import os



class FIDCalculator:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inception_model = inception_v3(weights=True, transform_input=False, aux_logits=True)
        self.inception_model.eval().to(self.device)

    def extract_features(self, images):
        # Check if images are grayscale (1 channel) and convert them to RGB (3 channels)
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)

        # Resize images if they are not 299 x 299
        if images.size(2) != 299 or images.size(3) != 299:
            images = torch.stack([TF.resize(img, (299, 299)) for img in images])

        # Normalize images
        normalized_images = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalized_images = normalized_images.to(self.device)

        with torch.no_grad():
            features = self.inception_model(normalized_images).detach().cpu().numpy()
        return features

    def calculate_statistics(self, features):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_fid(self, mu1, sigma1, mu2, sigma2):
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        # Check for imaginary numbers in covmean
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def get_fid_score(self, real_images, fake_images):
        # Check if images need resizing and resize if necessary
        if real_images.size(2) != 299 or real_images.size(3) != 299:
            real_images = torch.stack([TF.resize(img, (299, 299)) for img in real_images])
        if fake_images.size(2) != 299 or fake_images.size(3) != 299:
            fake_images = torch.stack([TF.resize(img, (299, 299)) for img in fake_images])

        real_features = self.extract_features(real_images)
        fake_features = self.extract_features(fake_images)

        mu_real, sigma_real = self.calculate_statistics(real_features)
        mu_fake, sigma_fake = self.calculate_statistics(fake_features)

        fid_score = self.calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
        return fid_score


def plot_fid_scores(epochs, fid_scores, save_dir):
    """
    Save a plot of FID scores over epochs to a specified directory.

    :param epochs: A list of epochs at which FID scores were calculated.
    :param fid_scores: A list of FID scores corresponding to the epochs.
    :param save_dir: Directory where the plot will be saved.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, fid_scores, marker='o', linestyle='-', color='b')
    plt.title('FID Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.grid(True)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    # Construct the file path
    file_path = os.path.join(save_dir, 'fid_scores_over_epochs.png')
    # Save the plot
    plt.savefig(file_path)
    plt.close()


def random_flip_tensors(images):
    """
    Randomly flip images horizontally and/or vertically with batch operations.

    Args:
    - images (Tensor): A batch of images with shape (B, C, H, W).

    Returns:
    - Tensor: The batch of images, randomly flipped.
    """
    # Generate random booleans for each image in the batch for horizontal and vertical flipping
    horizontal_flips = torch.rand(images.size(0), 1, 1, 1, device=images.device) < 0.5
    vertical_flips = torch.rand(images.size(0), 1, 1, 1, device=images.device) < 0.5

    # Apply horizontal flips
    images_horizontal_flipped = torch.where(horizontal_flips, images.flip(dims=[3]), images)

    # Apply vertical flips
    images_flipped = torch.where(vertical_flips, images_horizontal_flipped.flip(dims=[2]), images_horizontal_flipped)

    return images_flipped


def random_flip_tensors2(images):
    """
    Randomly flip images horizontally and/or vertically with batch operations.
    Also returns tensors indicating which images have been flipped.

    Args:
    - images (Tensor): A batch of images with shape (B, C, H, W).

    Returns:
    - Tensor: The batch of images, randomly flipped.
    - Tensor: Boolean tensor indicating which images were flipped horizontally.
    - Tensor: Boolean tensor indicating which images were flipped vertically.
    """
    # Generate random booleans for each image in the batch for horizontal and vertical flipping
    horizontal_flips = torch.rand(images.size(0), 1, 1, 1, device=images.device) < 0.5
    vertical_flips = torch.rand(images.size(0), 1, 1, 1, device=images.device) < 0.5

    # Prepare expanded flip tensors for broadcasting
    horizontal_flips_expanded = horizontal_flips.expand(-1, images.size(1), images.size(2), images.size(3))
    vertical_flips_expanded = vertical_flips.expand(-1, images.size(1), images.size(2), images.size(3))

    # Apply horizontal flips
    images_horizontal_flipped = torch.where(horizontal_flips_expanded, images.flip(dims=[3]), images)

    # Apply vertical flips
    images_flipped = torch.where(vertical_flips_expanded, images_horizontal_flipped.flip(dims=[2]), images_horizontal_flipped)

    # Collapse the flip indicator tensors to shape (B,)
    horizontal_flips = horizontal_flips.view(-1)
    vertical_flips = vertical_flips.view(-1)

    return images_flipped, horizontal_flips, vertical_flips


# To reverse the flipping for demonstration:
def reverse_flips(flipped_images, horizontal_flips, vertical_flips):
    # Reverse horizontal flips
    reversed_horizontal = torch.where(horizontal_flips.view(-1, 1, 1, 1).expand_as(flipped_images),
                                      flipped_images.flip(dims=[3]), flipped_images)

    # Reverse vertical flips
    reversed_images = torch.where(vertical_flips.view(-1, 1, 1, 1).expand_as(reversed_horizontal),
                                  reversed_horizontal.flip(dims=[2]), reversed_horizontal)

    return reversed_images



def show_images(real_image, fake_image, fake_fake_image, epoch, title, save_dir):
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 3, 1)
    plt.imshow(real_image[0].cpu().squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(fake_image[0].cpu().detach().squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(fake_fake_image[0].cpu().detach().squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(real_image[1].cpu().squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(fake_image[1].cpu().detach().squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(fake_fake_image[1].cpu().detach().squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.imshow(real_image[2].cpu().squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.imshow(fake_image[2].cpu().detach().squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.imshow(fake_fake_image[2].cpu().detach().squeeze(), cmap='gray')
    plt.axis('off')

    plt.savefig(save_dir + f"/{title}_{epoch:03d}.png")

    plt.close()


def show_images2(real_images, fake_images, fake_fake_images, fake_image_scores, fake_fake_image_scores,
                 epoch, title, save_dir):
    num_images = 5
    plt.figure(figsize=(15, 5 * num_images))

    for i in range(num_images):
        # Real Image
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(real_images[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Real Image")

        # Fake Image
        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(fake_images[i].cpu().detach().squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f"Score: {fake_image_scores[i].cpu().detach().item():.3f}")
        if i == 0:
            plt.title("Fake Image\n" + f"Score: {fake_image_scores[i].cpu().detach().item():.3f}")

        # Fake Fake Image
        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(fake_fake_images[i].cpu().detach().squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f"Score: {fake_fake_image_scores[i].cpu().detach().item():.3f}")
        if i == 0:
            plt.title("Fake Fake Image\n" + f"Score: {fake_fake_image_scores[i].cpu().detach().item():.3f}")

    plt.savefig(f"{save_dir}/{title}_{epoch:03d}.png")
    plt.close()

def plot_losses(losses_G_clean_to_noisy, losses_G_noisy_to_clean, losses_D_clean, losses_D_noisy, save_dir):
    plt.figure(figsize=(10, 5))
    plt.title("Training Losses During Training")
    plt.plot(losses_G_clean_to_noisy, label="Generator Clean to Noisy")
    plt.plot(losses_G_noisy_to_clean, label="Generator Noisy to Clean")
    plt.plot(losses_D_clean, label="Discriminator Clean")
    plt.plot(losses_D_noisy, label="Discriminator Noisy")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_dir + f"/loss.png")
    plt.close()


def plot_D_losses(D_clean, D_noisy, D_clean_MBD, D_noisy_MBD, cycle_loss_clean, cycle_loss_noisy, save_dir):
    plt.figure(figsize=(10, 5))
    plt.title("Training Discriminator Losses During Training")
    plt.plot(D_clean, label="D_clean")
    plt.plot(D_noisy, label="D_noisy")
    plt.plot(D_clean_MBD, label="D_clean_MBD")
    plt.plot(D_noisy_MBD, label="D_noisy_MBD")
    plt.plot(cycle_loss_clean, label="cycle_loss_clean")
    plt.plot(cycle_loss_noisy, label="cycle_loss_noisy")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_dir + f"/loss_D.png")
    plt.close()


def plot_D_losses2(D_clean, D_noisy, D_clean_MBD, D_noisy_MBD, cycle_loss_clean, cycle_loss_noisy, identity_clean,
                   identity_noisy,  save_dir):
    plt.figure(figsize=(10, 5))
    plt.title("Training Discriminator Losses During Training")
    plt.plot(D_clean, label="D_clean")
    plt.plot(D_noisy, label="D_noisy")
    plt.plot(D_clean_MBD, label="D_clean_MBD")
    plt.plot(D_noisy_MBD, label="D_noisy_MBD")
    plt.plot(cycle_loss_clean, label="cycle_loss_clean")
    plt.plot(cycle_loss_noisy, label="cycle_loss_noisy")
    plt.plot(identity_clean, label="identity_clean")
    plt.plot(identity_noisy, label="identity_noisy")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_dir + f"/loss_D.png")
    plt.close()



def visualize_progress(epoch, real_image, fake_image, losses_G_clean_to_noisy, losses_G_noisy_to_clean, losses_D_clean,
                       losses_D_noisy, title_prefix=""):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Display real and fake images
    axs[0].imshow(real_image.squeeze(), cmap='gray')
    axs[0].set_title("Real Image")
    axs[0].axis('off')

    axs[1].imshow(fake_image.squeeze(), cmap='gray')
    axs[1].set_title("Generated Image")
    axs[1].axis('off')

    # Plot losses
    axs[2].plot(losses_G_clean_to_noisy, label="G Clean to Noisy")
    axs[2].plot(losses_G_noisy_to_clean, label="G Noisy to Clean")
    axs[2].plot(losses_D_clean, label="D Clean")
    axs[2].plot(losses_D_noisy, label="D Noisy")
    axs[2].set_title("Loss over Epochs")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Loss")
    axs[2].legend()

    # Save figure
    fig.savefig(f"{title_prefix}progress_epoch_{epoch}.png")
    plt.close(fig)


def visualize_progress_simple(epoch, real_image, fake_image, title_prefix="", noisy_to_clean=False):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Display real and fake images
    axs[0].imshow(real_image[0].cpu().squeeze(), cmap='gray')
    if noisy_to_clean:
        axs[0].set_title("Real Noisy Image")
    else:
        axs[0].set_title("Real Clean Image")
    axs[0].axis('off')

    axs[1].imshow(fake_image[0].cpu().squeeze(), cmap='gray')
    if noisy_to_clean:
        axs[1].set_title("Generated Clean Image")
    else:
        axs[1].set_title("Generated Noisy Image")
    axs[1].axis('off')

    # Save figure
    if noisy_to_clean:
        fig.savefig(f"training_real005/video_nc/{title_prefix}_epoch_{epoch:03d}.png")
    else:
        fig.savefig(f"training_real005/video_cn/{title_prefix}_epoch_{epoch:03d}.png")
    plt.close(fig)


def adjust_learning_rate(optimizer, current_epoch, start_decay_epoch, end_decay_epoch, initial_lr):
    """
    Adjusts the learning rate linearly from its initial value to zero between start_decay_epoch and end_decay_epoch.

    Parameters:
    - optimizer: The optimizer for which to adjust the learning rate.
    - current_epoch: The current epoch number.
    - start_decay_epoch: The epoch from which to start decreasing the learning rate.
    - end_decay_epoch: The epoch by which the learning rate should reach zero.
    - initial_lr: The initial learning rate before decay starts.
    """
    if current_epoch < start_decay_epoch:
        lr = initial_lr
    elif current_epoch >= start_decay_epoch and current_epoch <= end_decay_epoch:
        lr = initial_lr * (1 - (current_epoch - start_decay_epoch) / (end_decay_epoch - start_decay_epoch))
    else:
        lr = 0

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_gradient_penalty(D, real_samples, fake_samples, center=1):
    # Calculate interpolation
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=real_samples.device)
    interpolated = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # Compute D(interpolated)
    d_interpolated = D(interpolated)

    # Compute gradients
    fake = torch.ones(d_interpolated.size(), requires_grad=False, device=real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Compute and return gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - center) ** 2).mean()
    return gradient_penalty
