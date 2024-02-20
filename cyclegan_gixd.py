import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import Adam
from tqdm import tqdm
import os
from datetime import datetime

from utensils import FIDCalculator, adjust_learning_rate, show_images, plot_losses
from cyclegan_networks import UNet2Conv, UNet2ConvRes, PatchGANDiscriminator, PatchGANDiscriminator32, MBDiscriminator
from image_simulation import (generate_2d_gaussian_image, generate_2d_gaussian_image2, generate_2d_gaussian_image3,
                              generate_2d_gaussian_image4, generate_noisy_image, generate_noisy_image2)


"""TRAINING PARAMETERS"""
save_folder = 'training_real006'  # folder where everything will be saved
real_images_np = np.load('image_slices_all_augmented_ekaterina_June_2020.npy')  # dataset (numpy) of real images

# for simulation
size = 128  # size of simulated images (do not change for now)
num_images = 2770  # number of images in dataset

# training parameters
batch_size = 8  # 10 for 2080ti, 30 for v100 are the limits (seems to have a lot of influence, 10 works best)
num_epochs = 400
start_decay_epoch = 350  # start and end of linear decay for the learning rate
end_decay_epoch = 400
initial_lr_G_clean_to_noisy = 0.01  # generator clean to noisy initial learning rate    0.002
initial_lr_G_noisy_to_clean = 0.01  # generator noisy to clean initial learning rate    0.005
initial_lr_D_clean = 0.0005  # discriminator for clean images initial learning rate     0.00005
initial_lr_D_noisy = 0.0005  # discriminator for noisy images initial learning rate     0.0002
pmbd_r = 0.7  # patchGAN discriminator to mini-batch discriminator influence ratio      0.8
alpha = 0.55  # The historical averaging coefficient                                    0.7 or 0.5 used to work good
lambda_cycle = 7  # Weight for cycle consistency loss                                   5 to 15

# Loss function
adversarial_loss = nn.BCELoss()
cycle_loss = nn.L1Loss()
identity_loss = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""CREATE DIRECTORY"""
current_time = datetime.now()
save_dir = (save_folder + "_" + str(current_time.day) + str(current_time.month) +
            str(current_time.hour) + str(current_time.minute))
os.mkdir(save_dir)

"""GENERATE SIMULATED IMAGES"""

print("Generate images")

clean_images = []
for _ in range(num_images):
    clean_image = generate_2d_gaussian_image4(size, norm2_1_1=False)  # create the images
    clean_images_min_noise = generate_noisy_image(clean_image, noise_level_low=0.03, noise_level_high=0.03,
                                                  norm2_1_1=True)  # add slight noise to promote backpropagation
    clean_images.append(clean_images_min_noise)
clean_images = np.array(clean_images)

"""DATASETS AND DATALOADERS"""

print("Prepare dataset")

# Convert datasets to PyTorch tensors
clean_images_tensor = torch.tensor(clean_images).unsqueeze(1).float()
noisy_images_tensor = torch.tensor(real_images_np).unsqueeze(1).float()

clean_dataset = TensorDataset(clean_images_tensor)
noisy_dataset = TensorDataset(noisy_images_tensor)
clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=True)
noisy_loader = DataLoader(noisy_dataset, batch_size=batch_size, shuffle=True)

""" Initialize the generator and discriminator """

print("Initialize the generator and discriminator")


"""NEURAL NETS"""
G_clean_to_noisy = UNet2ConvRes(n_channels=1, n_classes=1, num_residual_blocks=2)
G_noisy_to_clean = UNet2ConvRes(n_channels=1, n_classes=1, num_residual_blocks=2)

D_clean = PatchGANDiscriminator32(input_nc=1)
D_noisy = PatchGANDiscriminator32(input_nc=1)
D_clean_MBD = MBDiscriminator(input_nc=1, image_size=128)
D_noisy_MBD = MBDiscriminator(input_nc=1, image_size=128)


"""OPTIMIZERS"""
optimizer_G_clean_to_noisy = Adam(G_clean_to_noisy.parameters(), lr=initial_lr_G_clean_to_noisy, betas=(0.5, 0.999), weight_decay=0.0001)
optimizer_G_noisy_to_clean = Adam(G_noisy_to_clean.parameters(), lr=initial_lr_G_noisy_to_clean, betas=(0.5, 0.999), weight_decay=0.0001)

optimizer_D_clean = Adam(D_clean.parameters(), lr=initial_lr_D_clean, betas=(0.5, 0.999), weight_decay=0.0001)
optimizer_D_noisy = Adam(D_noisy.parameters(), lr=initial_lr_D_noisy, betas=(0.5, 0.999), weight_decay=0.0001)
optimizer_D_clean_MBD = Adam(D_clean_MBD.parameters(), lr=initial_lr_D_clean, betas=(0.5, 0.999), weight_decay=0.0001)
optimizer_D_noisy_MBD = Adam(D_noisy_MBD.parameters(), lr=initial_lr_D_noisy, betas=(0.5, 0.999), weight_decay=0.0001)

""" TRAINING LOOP """

print("Start training")

G_clean_to_noisy.to(device)
G_noisy_to_clean.to(device)
D_clean.to(device)
D_noisy.to(device)
D_clean_MBD.to(device)
D_noisy_MBD.to(device)

# Initialize lists to store losses for each generator and discriminator
g_losses = []
d_losses = []
losses_G_clean_to_noisy = []
losses_G_noisy_to_clean = []
losses_D_clean = []
losses_D_noisy = []

"""HISTORICAL AVERAGE"""
# Initial historical average
historical_params_G_clean_to_noisy = {name: p.clone().detach() for name, p in G_clean_to_noisy.named_parameters()}
historical_params_G_noisy_to_clean = {name: p.clone().detach() for name, p in G_noisy_to_clean.named_parameters()}
historical_params_D_clean = {name: p.clone().detach() for name, p in D_clean.named_parameters()}
historical_params_D_noisy = {name: p.clone().detach() for name, p in D_noisy.named_parameters()}
historical_params_D_clean_MBD = {name: p.clone().detach() for name, p in D_clean_MBD.named_parameters()}
historical_params_D_noisy_MBD = {name: p.clone().detach() for name, p in D_noisy_MBD.named_parameters()}

for epoch in range(num_epochs):

    # Adjust learning rates
    adjust_learning_rate(optimizer_G_clean_to_noisy, epoch, start_decay_epoch, end_decay_epoch,
                         initial_lr_G_clean_to_noisy)
    adjust_learning_rate(optimizer_G_noisy_to_clean, epoch, start_decay_epoch, end_decay_epoch,
                         initial_lr_G_noisy_to_clean)
    adjust_learning_rate(optimizer_D_clean, epoch, start_decay_epoch, end_decay_epoch, initial_lr_D_clean)
    adjust_learning_rate(optimizer_D_noisy, epoch, start_decay_epoch, end_decay_epoch, initial_lr_D_noisy)
    adjust_learning_rate(optimizer_D_clean_MBD, epoch, start_decay_epoch, end_decay_epoch, initial_lr_D_clean)
    adjust_learning_rate(optimizer_D_noisy_MBD, epoch, start_decay_epoch, end_decay_epoch, initial_lr_D_noisy)

    for i, (clean_data, noisy_data) in enumerate(zip(clean_loader, noisy_loader)):
        real_clean = clean_data[0].to(device)
        real_noisy = noisy_data[0].to(device)

        # for simple discriminator
        valid_MBD = torch.ones(real_clean.size(0), 1, device=device)
        fake_MBD = torch.zeros(real_clean.size(0), 1, device=device)

        # for Patch Discriminator
        patch_size = 31  # 7 
        num_patches = patch_size * patch_size
        valid = torch.ones(real_clean.size(0), 1, patch_size, patch_size, device=device)
        fake = torch.zeros(real_clean.size(0), 1, patch_size, patch_size, device=device)

        # ------------------------
        #  Train Generators
        # ------------------------

        optimizer_G_clean_to_noisy.zero_grad()
        optimizer_G_noisy_to_clean.zero_grad()

        # Generate a batch of "fake" images
        fake_noisy = G_clean_to_noisy(real_clean)
        fake_clean = G_noisy_to_clean(real_noisy)

        # Adversarial loss
        loss_GAN_clean_to_noisy = adversarial_loss(D_noisy(fake_noisy), valid)
        loss_GAN_noisy_to_clean = adversarial_loss(D_clean(fake_clean), valid)

        loss_GAN_clean_to_noisy_MBD = adversarial_loss(D_noisy_MBD(fake_noisy), valid_MBD)
        loss_GAN_noisy_to_clean_MBD = adversarial_loss(D_clean_MBD(fake_clean), valid_MBD)

        # Cycle loss
        recon_clean = G_noisy_to_clean(fake_noisy)
        recon_noisy = G_clean_to_noisy(fake_clean)
        loss_cycle_clean = cycle_loss(recon_clean, real_clean) * lambda_cycle
        loss_cycle_noisy = cycle_loss(recon_noisy, real_noisy) * lambda_cycle

        # Identity loss
        identity_clean = G_noisy_to_clean(real_clean)
        identity_noisy = G_clean_to_noisy(recon_noisy)

        identity_loss_G_clean = identity_loss(real_clean, identity_clean)
        identity_loss_G_noisy = identity_loss(recon_noisy, identity_noisy)

        # Total generator losses
        loss_G_clean = (loss_cycle_clean + identity_loss_G_clean +
                        (loss_GAN_clean_to_noisy * pmbd_r + loss_GAN_clean_to_noisy_MBD * (1 - pmbd_r)))
        loss_G_noisy = (loss_cycle_noisy + identity_loss_G_noisy +
                        (loss_GAN_noisy_to_clean * pmbd_r + loss_GAN_noisy_to_clean_MBD * (1 - pmbd_r)))

        # Total generator loss
        loss_G = (loss_cycle_clean + loss_cycle_noisy + identity_loss_G_clean + identity_loss_G_noisy +
                  (loss_GAN_clean_to_noisy + loss_GAN_noisy_to_clean) * pmbd_r +
                  (loss_GAN_clean_to_noisy_MBD + loss_GAN_clean_to_noisy_MBD) * (1 - pmbd_r))

        loss_G.backward()

        optimizer_G_clean_to_noisy.step()
        optimizer_G_noisy_to_clean.step()

        # ------------------------
        #  Train Discriminators
        # ------------------------

        optimizer_D_clean.zero_grad()
        optimizer_D_noisy.zero_grad()
        optimizer_D_clean_MBD.zero_grad()
        optimizer_D_noisy_MBD.zero_grad()

        # Real loss
        loss_real_clean = adversarial_loss(D_clean(real_clean), valid)
        loss_real_noisy = adversarial_loss(D_noisy(real_noisy), valid)

        loss_real_clean_MBD = adversarial_loss(D_clean_MBD(real_clean), valid_MBD)
        loss_real_noisy_MBD = adversarial_loss(D_noisy_MBD(real_noisy), valid_MBD)

        # Fake loss
        loss_fake_clean = adversarial_loss(D_clean(fake_clean.detach()), fake)
        loss_fake_noisy = adversarial_loss(D_noisy(fake_noisy.detach()), fake)

        loss_fake_clean_MBD = adversarial_loss(D_clean_MBD(fake_clean.detach()), fake_MBD)
        loss_fake_noisy_MBD = adversarial_loss(D_noisy_MBD(fake_noisy.detach()), fake_MBD)

        loss_D_clean = ((loss_real_clean + loss_fake_clean) * pmbd_r + (loss_real_clean_MBD + loss_fake_clean_MBD) * (1 - pmbd_r)) / 2
        loss_D_noisy = ((loss_real_noisy + loss_fake_noisy) * pmbd_r + (loss_real_noisy_MBD + loss_fake_noisy_MBD) * (1 - pmbd_r)) / 2

        loss_D_clean.backward()
        loss_D_noisy.backward()

        optimizer_D_clean.step()
        optimizer_D_noisy.step()
        
        optimizer_D_clean_MBD.step()
        optimizer_D_noisy_MBD.step()

        # ------------------------
        #  Adjust historical average
        # ------------------------

        for name, param in G_clean_to_noisy.named_parameters():
            historical_params_G_clean_to_noisy[name].mul_(alpha).add_(param.data, alpha=1 - alpha)
            param.data.copy_(historical_params_G_clean_to_noisy[name])

        for name, param in G_noisy_to_clean.named_parameters():
            historical_params_G_noisy_to_clean[name].mul_(alpha).add_(param.data, alpha=1 - alpha)
            param.data.copy_(historical_params_G_noisy_to_clean[name])

        for name, param in D_clean.named_parameters():
            historical_params_D_clean[name].mul_(alpha).add_(param.data, alpha=1 - alpha)
            param.data.copy_(historical_params_D_clean[name])

        for name, param in D_noisy.named_parameters():
            historical_params_D_noisy[name].mul_(alpha).add_(param.data, alpha=1 - alpha)
            param.data.copy_(historical_params_D_noisy[name])

        for name, param in D_clean_MBD.named_parameters():
            historical_params_D_clean_MBD[name].mul_(alpha).add_(param.data, alpha=1 - alpha)
            param.data.copy_(historical_params_D_clean_MBD[name])

        for name, param in D_noisy_MBD.named_parameters():
            historical_params_D_noisy_MBD[name].mul_(alpha).add_(param.data, alpha=1 - alpha)
            param.data.copy_(historical_params_D_noisy_MBD[name])

        # ------------------------
        #  Save images and model
        # ------------------------

        if (epoch % 5 == 0) and i == 0:  # Optionally display images every 5 epochs
            with torch.no_grad():
                generated_images_noisy = G_clean_to_noisy(real_clean)
                generated_generated_images = G_noisy_to_clean(generated_images_noisy)
                show_images(real_clean, generated_images_noisy, generated_generated_images, epoch, "clean", save_dir)

                generated_images = G_noisy_to_clean(real_noisy)
                generated_generated_images = G_clean_to_noisy(generated_images)
                show_images(real_noisy, generated_images, generated_generated_images, epoch, "noisy", save_dir)

        # if (epoch % 50 == 0) and i == 0:

            # Calculate FID Score
            # fid_score = fid_calculator.get_fid_score(real_noisy, generated_images_noisy)
            # print(f"FID score: {fid_score}")

        if (epoch % 50 == 0) and (i == 0):
            # Save model weights
            torch.save(G_clean_to_noisy.state_dict(), save_dir + '/G_clean_to_noisy_weights_' + str(epoch) + '.pth')
            torch.save(G_noisy_to_clean.state_dict(), save_dir + '/G_noisy_to_clean_weights_' + str(epoch) + '.pth')
            torch.save(D_clean.state_dict(), save_dir + '/D_clean_weights_' + str(epoch) + '.pth')
            torch.save(D_noisy.state_dict(), save_dir + '/D_noisy_weights_' + str(epoch) + '.pth')
            torch.save(D_clean_MBD.state_dict(), save_dir + '/D_clean_MBD_weights_' + str(epoch) + '.pth')
            torch.save(D_noisy_MBD.state_dict(), save_dir + '/D_noisy_MBD_weights_' + str(epoch) + '.pth')

            # Save entire models
            torch.save(G_clean_to_noisy, save_dir + '/G_clean_to_noisy_' + str(epoch) + '.pth')
            torch.save(G_noisy_to_clean, save_dir + '/G_noisy_to_clean_' + str(epoch) + '.pth')
            torch.save(D_clean, save_dir + '/D_clean_' + str(epoch) + '.pth')
            torch.save(D_noisy, save_dir + '/D_noisy_' + str(epoch) + '.pth')
            torch.save(D_clean_MBD, save_dir + '/D_clean_MBD_' + str(epoch) + '.pth')
            torch.save(D_noisy_MBD, save_dir + '/D_noisy_MBD_' + str(epoch) + '.pth')

    losses_G_clean_to_noisy.append(loss_GAN_clean_to_noisy.item())
    losses_G_noisy_to_clean.append(loss_GAN_noisy_to_clean.item())
    losses_D_clean.append(loss_D_clean.item())
    losses_D_noisy.append(loss_D_noisy.item())

    # print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss D_clean: {loss_D_clean.item():.5f}, Loss D_noisy: {loss_D_noisy.item():.5f}, Loss G: {loss_G.item():.5f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss D_clean: {loss_D_clean.item():.5f},"
          f" Loss D_noisy: {loss_D_noisy.item():.5f}, Loss G_Clean_to_Noisy: {loss_G_clean.item():.5f}, Loss G_Noisy_to_Clean: {loss_G_noisy.item():.5f}")

    plot_losses(losses_G_clean_to_noisy, losses_G_noisy_to_clean, losses_D_clean, losses_D_noisy, save_dir)

# Save model weights
torch.save(G_clean_to_noisy.state_dict(), save_dir + '/G_clean_to_noisy_weights.pth')
torch.save(G_noisy_to_clean.state_dict(), save_dir +'/G_noisy_to_clean_weights.pth')
torch.save(D_clean.state_dict(), save_dir + '/D_clean_weights.pth')
torch.save(D_noisy.state_dict(), save_dir + '/D_noisy_weights.pth')
torch.save(D_clean_MBD.state_dict(), save_dir + '/D_clean_MBD_weights.pth')
torch.save(D_noisy_MBD.state_dict(), save_dir + '/D_noisy_MBD_weights.pth')

# Save entire models
torch.save(G_clean_to_noisy, save_dir + '/G_clean_to_noisy.pth')
torch.save(G_noisy_to_clean, save_dir + '/G_noisy_to_clean.pth')
torch.save(D_clean, save_dir + '/D_clean.pth')
torch.save(D_noisy, save_dir + '/D_noisy.pth')
torch.save(D_clean_MBD, save_dir + '/D_clean_MBD.pth')
torch.save(D_noisy_MBD, save_dir + '/D_noisy_MBD.pth')
