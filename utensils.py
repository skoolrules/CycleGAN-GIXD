from scipy.linalg import sqrtm
import numpy as np
import torch
from torchvision.models import inception_v3
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

class FIDCalculator:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inception_model = inception_v3(pretrained=True, transform_input=False, aux_logits=False)
        self.inception_model.eval().to(self.device)

    def extract_features(self, images):
        # Resize and normalize images for InceptionV3
        images = torch.stack([TF.resize(image, (299, 299)) for image in images])
        images = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images = images.to(self.device)

        with torch.no_grad():
            features = self.inception_model(images).detach().cpu().numpy()
        return features

    def calculate_statistics(self, features):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_fid(self, mu1, sigma1, mu2, sigma2):
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2), disp=False)[0]
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def get_fid_score(self, real_images, fake_images):
        real_features = self.extract_features(real_images)
        fake_features = self.extract_features(fake_images)

        mu_real, sigma_real = self.calculate_statistics(real_features)
        mu_fake, sigma_fake = self.calculate_statistics(fake_features)

        fid_score = self.calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
        return fid_score


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
    plt.savefig(save_dir + f"loss.png")
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

