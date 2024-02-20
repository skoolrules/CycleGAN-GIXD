import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import Adam
import os

def generate_2d_gaussian_image(image_size, mean=None, amplitude_range=(0.5, 1.0), sigma_x_range=(0.05, 0.6),
                               sigma_y_range=(0.05, 0.6)):
    """
    Generate an image with a single 2D Gaussian distribution.

    Parameters:
    - image_size (int): Size of the image (image will be square).
    - mean (tuple of floats): Mean (center) of the Gaussian. If None, a random mean is chosen.
    - amplitude_range (tuple of floats): Range of possible amplitudes (peak heights) for the Gaussian.
    - sigma_x_range (tuple of floats): Range of possible standard deviations along the x-axis.
    - sigma_y_range (tuple of floats): Range of possible standard deviations along the y-axis.

    Returns:
    - 2D numpy array representing the image.
    """
    # Create a grid of (x,y) coordinates
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    x, y = np.meshgrid(x, y)

    # If mean is not specified, choose a random mean
    if mean is None:
        mean = np.random.uniform(-1, 1, size=2)

    # Randomly select amplitude, sigma_x, and sigma_y values within the specified ranges
    amplitude = np.random.uniform(*amplitude_range)
    sigma_x = np.random.uniform(*sigma_x_range)
    sigma_y = np.random.uniform(*sigma_y_range)

    # Calculate the Gaussian
    gaussian = amplitude * np.exp(-((x - mean[0]) ** 2 / (2 * sigma_x ** 2) + (y - mean[1]) ** 2 / (2 * sigma_y ** 2)))

    return gaussian


def generate_2d_gaussian_image2(image_size, num_gaussians_range=(1, 3), amplitude_range=(0.5, 1.0),
                                sigma_x_range=(0.05, 0.3), sigma_y_range=(0.05, 1.5), min_distance=0.4):
    """
    Generate an image with a random number of 2D Gaussian distributions, ensuring Gaussians are not too close to each other.

    Parameters:
    - image_size (int): Size of the image (image will be square).
    - num_gaussians_range (tuple of ints): Range (min, max) of the number of Gaussian distributions to generate in the image.
    - amplitude_range (tuple of floats): Range of possible amplitudes (peak heights) for each Gaussian.
    - sigma_x_range (tuple of floats): Range of possible standard deviations along the x-axis for each Gaussian.
    - sigma_y_range (tuple of floats): Range of possible standard deviations along the y-axis for each Gaussian.
    - min_distance (float): Minimum distance between the centers of different Gaussians.

    Returns:
    - 2D numpy array representing the image.
    """
    # Create a grid of (x,y) coordinates
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    x, y = np.meshgrid(x, y)

    # Initialize the image with zeros
    image = np.zeros((image_size, image_size))

    # Randomly select the number of Gaussians to generate
    num_gaussians = np.random.randint(*num_gaussians_range)
    gaussian_means = []

    for _ in range(num_gaussians):
        too_close = True
        while too_close:
            # Randomly select mean
            mean = np.random.uniform(-0.85, 0.85, size=2)
            too_close = any(np.linalg.norm(mean - existing_mean) < min_distance for existing_mean in gaussian_means)

        gaussian_means.append(mean)
        amplitude = np.random.uniform(*amplitude_range)
        sigma_x = np.random.uniform(*sigma_x_range)
        sigma_y = np.random.uniform(*sigma_y_range)

        # Calculate the Gaussian and add it to the image
        gaussian = amplitude * np.exp(
            -((x - mean[0]) ** 2 / (2 * sigma_x ** 2) + (y - mean[1]) ** 2 / (2 * sigma_y ** 2)))
        image += gaussian

    return image


def generate_2d_gaussian_image3(image_size, num_gaussians_range=(1, 5), amplitude_range=(0.2, 1.0),
                                sigma_x_range=(0.01, 0.08), sigma_y_range=(0.05, 5), c_range=(0.05, 0.2),
                                min_distance=0.4, norm2_1_1=False):
    """
    Generate an image with a random number of 2D Gaussian distributions, ensuring Gaussians are not too close to each other,
    with an added offset c to each Gaussian distribution, and normalize the image to be in the range 0 to 1.

    Parameters:
    - image_size (int): Size of the image (image will be square).
    - num_gaussians_range (tuple of ints): Range (min, max) of the number of Gaussian distributions to generate in the image.
    - amplitude_range (tuple of floats): Range of possible amplitudes (peak heights) for each Gaussian.
    - sigma_x_range (tuple of floats): Range of possible standard deviations along the x-axis for each Gaussian.
    - sigma_y_range (tuple of floats): Range of possible standard deviations along the y-axis for each Gaussian.
    - c_range (tuple of floats): Range of possible offsets for each Gaussian.
    - min_distance (float): Minimum distance between the centers of different Gaussians.

    Returns:
    - 2D numpy array representing the normalized image.
    """
    # Create a grid of (x,y) coordinates
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    x, y = np.meshgrid(x, y)

    # Initialize the image with zeros
    image = np.zeros((image_size, image_size))

    # Randomly select the number of Gaussians to generate
    num_gaussians = np.random.randint(*num_gaussians_range)
    gaussian_means = []

    for _ in range(num_gaussians):
        too_close = True
        while too_close:
            # Randomly select mean
            mean = np.random.uniform(-0.85, 0.85, size=2)
            too_close = any(np.linalg.norm(mean - existing_mean) < min_distance for existing_mean in gaussian_means)

        gaussian_means.append(mean)
        amplitude = np.random.uniform(*amplitude_range)
        sigma_x = np.random.uniform(*sigma_x_range)
        sigma_y = np.random.uniform(*sigma_y_range)
        c = np.random.uniform(*c_range)  # Randomly select offset c

        # Calculate the Gaussian with offset c and add it to the image
        gaussian = amplitude * np.exp(-((x - mean[0]) ** 2 / (2 * sigma_x ** 2) + (y - mean[1]) ** 2 / (2 * sigma_y ** 2))) + c
        image += gaussian

    if norm2_1_1:
        # Normalize the image to be in the range [-1, 1]
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min) * 2 - 1

    else:
        # Normalize the image to be in the range [0, 1]
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min)

    return image


def generate_2d_gaussian_image4(image_size, num_gaussians_range=(0, 8), amplitude_range=(0.5, 1.0),
                                sigma_x_range=(0.01, 0.06), sigma_y_range=(0.5, 5), c_range=(0.05, 0.2),
                                min_distance=0.5, norm2_1_1=False, rotation_angle_range=(-0.03, 0.03),
                                rotation_probability=0.3):
    """
    Generate an image with a random number of rotated 2D Gaussian distributions.
    """
    # Create a grid of (x,y) coordinates
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    x, y = np.meshgrid(x, y)

    # Initialize the image with zeros
    image = np.zeros((image_size, image_size))

    # Randomly select the number of Gaussians to generate
    num_gaussians = np.random.randint(*num_gaussians_range)
    gaussian_means = []

    if num_gaussians == 0:
        image = np.ones((image_size, image_size))

    else:
        for _ in range(num_gaussians):
            too_close = True
            while too_close:
                # Randomly select mean
                mean = np.random.uniform(-1, 1, size=2)
                too_close = any(np.linalg.norm(mean - existing_mean) < min_distance for existing_mean in gaussian_means)

            gaussian_means.append(mean)
            amplitude = np.random.uniform(*amplitude_range)
            sigma_x = np.random.uniform(*sigma_x_range)
            sigma_y = np.random.uniform(*sigma_y_range)
            c = np.random.uniform(*c_range)  # Randomly select offset c

            # Decide whether to rotate and calculate rotation angle
            if np.random.rand() < rotation_probability:
                theta = np.random.uniform(*rotation_angle_range)
            else:
                theta = 0  # No rotation

            # Adjust the coordinates based on the rotation angle
            x_rot = (x - mean[0]) * np.cos(theta) + (y - mean[1]) * np.sin(theta)
            y_rot = -(x - mean[0]) * np.sin(theta) + (y - mean[1]) * np.cos(theta)

            # Calculate the rotated Gaussian with offset c
            gaussian = amplitude * np.exp(-((x_rot) ** 2 / (2 * sigma_x ** 2) + (y_rot) ** 2 / (2 * sigma_y ** 2))) + c
            image += gaussian

        if norm2_1_1:
            # Normalize the image to be in the range [-1, 1]
            image_min = image.min()
            image_max = image.max()
            image = (image - image_min) / (image_max - image_min) * 2 - 1
        else:
            # Normalize the image to be in the range [0, 1]
            image_min = image.min()
            image_max = image.max()
            image = (image - image_min) / (image_max - image_min)

    return image


def generate_noisy_image(gaussian, noise_level_low=0.0001, noise_level_high=0.3, norm2_1_1=False):
    noise_level = np.random.uniform(noise_level_low, noise_level_high)
    noise = np.random.normal(0.5, noise_level, gaussian.shape)
    noisy_image = gaussian + noise

    if norm2_1_1:
        # Normalize the image to be in the range [0, 1]
        image_min = noisy_image.min()
        image_max = noisy_image.max()
        noisy_image = (noisy_image - image_min) / (image_max - image_min) * 2 - 1

        noisy_image = np.clip(noisy_image, -1, 1)

    else:
        # Normalize the image to be in the range [0, 1]
        image_min = noisy_image.min()
        image_max = noisy_image.max()
        noisy_image = (noisy_image - image_min) / (image_max - image_min)

        noisy_image = np.clip(noisy_image, 0.01, 1)
    return noisy_image


def generate_noisy_image2(gaussian, noise_level_low=0.0001, noise_level_high=0.3, poisson_scale=10, norm2_1_1=False):
    """
    Generate a noisy image by adding Gaussian and Poisson noise to a Gaussian distribution image.

    Parameters:
    - gaussian (2D numpy array): The original image with Gaussian distributions.
    - noise_level_low (float): Lower bound for the standard deviation of Gaussian noise.
    - noise_level_high (float): Upper bound for the standard deviation of Gaussian noise.
    - poisson_scale (float): Scaling factor for Poisson noise intensity.

    Returns:
    - 2D numpy array representing the noisy image.
    """
    # Gaussian noise
    noise_level = np.random.uniform(noise_level_low, noise_level_high)
    gaussian_noise = np.random.normal(0.5, noise_level, gaussian.shape)

    # Poisson noise
    # Scale the image to a suitable range for Poisson noise
    scaled_image = gaussian * poisson_scale
    poisson_noise = np.random.poisson(scaled_image) / poisson_scale - gaussian

    # Combine the original image with both types of noise
    noisy_image = gaussian + gaussian_noise + poisson_noise

    if norm2_1_1:
        # Clip the image to maintain pixel value range
        noisy_image = np.clip(noisy_image, -1, 1)

    else:
        # Clip the image to maintain pixel value range
        noisy_image = np.clip(noisy_image, 0.01, 1)

    return noisy_image
