import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
import os
from datetime import datetime
import json

# import modules
from utensils import (FIDCalculator, plot_fid_scores, adjust_learning_rate, show_images, show_images2, plot_losses,
                      plot_D_losses, plot_D_losses2,
                      random_flip_tensors, compute_gradient_penalty)
from cyclegan_networks import (UNet2Conv, UNet2ConvRes, PatchGANDiscriminator, PatchGANDiscriminator32, MBDiscriminator,
                               CombinedDiscriminator, visualize_realness_map)
from image_simulation import (generate_2d_gaussian_image, generate_2d_gaussian_image2, generate_2d_gaussian_image3,
                              generate_2d_gaussian_image4, generate_noisy_image, generate_noisy_image2,
                              apply_random_mask,
                              generate_2d_gaussian_image_with_noise_and_boxes)

# torch.autograd.set_detect_anomaly(True)

"""LOAD CONFIG FILE"""
# Load the configurations
with open('init_config.json') as config_file:
    config = json.load(config_file)

"""CREATE DIRECTORY AND LOAD TRAINING IMAGES"""
save_folder = 'training_real008'  # folder where everything will be saved

real_images_np = np.load('image_slices_all_augmented_ekaterina_June_2020.npy')  # dataset (numpy) of real images
# real_images_np = np.load('image_slices_mega.npy')  # dataset (numpy) of real images

image_masks = np.load('image_slices_mask.npy')

current_time = datetime.now()
save_dir = (save_folder + "_" + str(current_time.day) + "_" + str(current_time.month) + "_" +
            str(current_time.hour) + "_" + str(current_time.minute))
os.mkdir(save_dir)

"""TRAINING PARAMETERS"""
# for simulation
size = 128  # size of simulated images (do not change for now)
num_images = real_images_np.shape[0]  # 2770  # number of images in dataset

use_hard_coded_config = True
if use_hard_coded_config:
    # training parameters
    batch_size = 45  # 8 for 2080ti, 30 for v100, 45 for a100 are the limits (seems to have a lot of influence, 10 works best?)
    num_epochs = 801
    start_decay_epoch = 400  # start and end of linear decay for the learning rate
    end_decay_epoch = 800
    initial_lr_G_clean_to_noisy = 0.001  # generator clean to noisy initial learning rate    0.002 0.01                 0.0001
    initial_lr_G_noisy_to_clean = 0.001  # generator noisy to clean initial learning rate    0.005                      0.0001
    initial_lr_D_clean = 0.001  # discriminator for clean images initial learning rate     0.00005 0.0005               for MBD 0.0001
    initial_lr_D_noisy = 0.001  # discriminator for noisy images initial learning rate     0.0002                       for MBD 0.0001
    initial_lr_Ds_MBD = 0.0001    # patchGAN discriminator to mini-batch discriminator learning rate ratio                0.0001
    alpha_G = 0.9
    alpha = 0.5  # The historical averaging coefficient                                    0.7 or 0.5 used to work good 0.5
    lambda_cycle = 10  # Weight for cycle consistency loss                                   5 to 15
    lambda_gp = 0.1  # Weight for gradient penalty loss      0.01 was okay                   # maybe 0.1???????
    l_constraint = 0

    # choose if to include identity loss
    use_identity_loss = True  # using identity loss makes the learning rate "more sensitive"
    use_mbd = True
    augment_training_data = True
    generate_new_simulated_images_each_epoch = False


else:
    # training parameters
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    start_decay_epoch = config['start_decay_epoch']
    end_decay_epoch = config['end_decay_epoch']
    initial_lr_G_clean_to_noisy = config['initial_lr_G_clean_to_noisy']
    initial_lr_G_noisy_to_clean = config['initial_lr_G_noisy_to_clean']
    initial_lr_D_clean = config['initial_lr_D_clean']
    initial_lr_D_noisy = config['initial_lr_D_noisy']
    initial_lr_Ds_MBD = config['initial_lr_Ds_MBD']
    pmbd_r = config['pmbd_r']
    alpha = config['alpha']
    lambda_cycle = config['lambda_cycle']
    lambda_gp = config['lambda_gp']

    # choose if to include identity loss
    use_identity_loss = config['use_identity_loss']
    use_mbd = config['use_mbd']
    augment_training_data = config['augment_training_data']

    # save config file for reproducibility
    config_save_path = os.path.join(save_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)

# Loss function
adversarial_loss = nn.BCELoss()  # nn.MSELoss()
cycle_loss = nn.L1Loss()
identity_loss = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

num_gpus = torch.cuda.device_count()
print(f"{num_gpus} GPUs available")

"""GENERATE SIMULATED IMAGES"""

print("Generate images")

clean_images = []
for _ in range(num_images):
    clean_image, boxes = generate_2d_gaussian_image_with_noise_and_boxes(size, norm2_1_1=True)
    clean_images.append(clean_image)
clean_images = np.array(clean_images)
clean_images = apply_random_mask(clean_images, image_masks)

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
G_clean_to_noisy = (UNet2ConvRes(n_channels=1, n_classes=1, num_residual_blocks=10))  # num_residual_blocks=10
G_noisy_to_clean = (UNet2ConvRes(n_channels=1, n_classes=1, num_residual_blocks=10))

D_clean = (PatchGANDiscriminator32(input_nc=1))
D_noisy = (PatchGANDiscriminator32(input_nc=1))
if use_mbd:
    D_clean_MBD = (MBDiscriminator(input_nc=1, image_size=128))
    D_noisy_MBD = (MBDiscriminator(input_nc=1, image_size=128))

G_clean_to_noisy.to(device)
G_noisy_to_clean.to(device)
D_clean.to(device)
D_noisy.to(device)

if use_mbd:
    D_clean_MBD.to(device)
    D_noisy_MBD.to(device)

"""LOAD PRE-TRAINED NETWORKS"""

use_pre_trained_generators = False

if use_pre_trained_generators:
    print("Loading pretrained Generators")
    model_path0 = "G_pretrained_resnet10_background_tanh.pth"
    G_clean_to_noisy.load_state_dict(torch.load(model_path0))
    model_path1 = "G_pretrained_resnet10_background_tanh.pth"
    G_noisy_to_clean.load_state_dict(torch.load(model_path1))

"""LOAD ALREADY TRAINED NETWORKS"""

use_trained_models = False

if use_trained_models:
    print("Loading Models")
    # Generators
    model_path0 = "training_real007_1_3_10_22_succesful/G_clean_to_noisy_weights.pth"
    G_clean_to_noisy.load_state_dict(torch.load(model_path0))
    model_path1 = "training_real007_1_3_10_22_succesful/G_noisy_to_clean_weights.pth"
    G_noisy_to_clean.load_state_dict(torch.load(model_path1))

    # Patch Discriminators
    model_path2 = "training_real007_1_3_10_22_succesful/D_clean_weights.pth"
    D_clean.load_state_dict(torch.load(model_path2))
    model_path3 = "training_real007_1_3_10_22_succesful/D_noisy_weights.pth"
    D_noisy.load_state_dict(torch.load(model_path3))

    # Minibatch Discriminators
    model_path4 = "training_real007_1_3_10_22_succesful/D_clean_MBD_weights.pth"
    D_clean_MBD.load_state_dict(torch.load(model_path4))
    model_path5 = "training_real007_1_3_10_22_succesful/D_noisy_MBD_weights.pth"
    D_noisy_MBD.load_state_dict(torch.load(model_path5))

"""OPTIMIZERS"""
optimizer_G_clean_to_noisy = Adam(G_clean_to_noisy.parameters(), lr=initial_lr_G_clean_to_noisy, betas=(0.5, 0.999),
                                  weight_decay=0.0001)
optimizer_G_noisy_to_clean = Adam(G_noisy_to_clean.parameters(), lr=initial_lr_G_noisy_to_clean, betas=(0.5, 0.999),
                                  weight_decay=0.0001)

optimizer_D_clean = Adam(D_clean.parameters(), lr=initial_lr_D_clean, betas=(0.5, 0.999), weight_decay=0.0001)
optimizer_D_noisy = Adam(D_noisy.parameters(), lr=initial_lr_D_noisy, betas=(0.5, 0.999), weight_decay=0.0001)
if use_mbd:
    optimizer_D_clean_MBD = Adam(D_clean_MBD.parameters(), lr=initial_lr_Ds_MBD, betas=(0.5, 0.999),
                                 weight_decay=0.0001)
    optimizer_D_noisy_MBD = Adam(D_noisy_MBD.parameters(), lr=initial_lr_Ds_MBD, betas=(0.5, 0.999),
                                 weight_decay=0.0001)

"""EVALUATION METRICS"""
fid_calculator = FIDCalculator(device=device)

""" TRAINING LOOP """

print("Start training")

# Initialize lists to store losses for each generator and discriminator
g_losses = []
d_losses = []
losses_G_clean_to_noisy = []
losses_G_noisy_to_clean = []
losses_D_clean = []
losses_D_noisy = []

losses_D_clean_patch = []
losses_D_noisy_patch = []
losses_D_clean_MBD = []
losses_D_noisy_MBD = []

losses_cycle_clean = []
losses_cycle_noisy = []

losses_identity_clean = []
losses_identity_noisy = []

fid_score_list = []
fid_epochs = []

real_features = []
fake_features = []

ree = False

"""HISTORICAL AVERAGE"""
# Initial historical average
historical_params_G_clean_to_noisy = {name: p.clone().detach() for name, p in G_clean_to_noisy.named_parameters()}
historical_params_G_noisy_to_clean = {name: p.clone().detach() for name, p in G_noisy_to_clean.named_parameters()}
historical_params_D_clean = {name: p.clone().detach() for name, p in D_clean.named_parameters()}
historical_params_D_noisy = {name: p.clone().detach() for name, p in D_noisy.named_parameters()}

if use_mbd:
    historical_params_D_clean_MBD = {name: p.clone().detach() for name, p in D_clean_MBD.named_parameters()}
    historical_params_D_noisy_MBD = {name: p.clone().detach() for name, p in D_noisy_MBD.named_parameters()}

for epoch in range(num_epochs):

    # Reset features lists at the start of each epoch
    real_features.clear()
    fake_features.clear()

    if (epoch % 25 == 0) and (epoch != 0):
        print("regenerate dataset")
        clean_images = []
        for _ in range(num_images):
            clean_image, boxes = generate_2d_gaussian_image_with_noise_and_boxes(size, norm2_1_1=True)
            clean_images.append(clean_image)
        clean_images = np.array(clean_images)
        clean_images = apply_random_mask(clean_images, image_masks)
        clean_images_tensor = torch.tensor(clean_images).unsqueeze(1).float()
        clean_dataset = TensorDataset(clean_images_tensor)
        clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=True)

    # Adjust learning rates
    adjust_learning_rate(optimizer_G_clean_to_noisy, epoch, start_decay_epoch, end_decay_epoch,
                         initial_lr_G_clean_to_noisy)
    adjust_learning_rate(optimizer_G_noisy_to_clean, epoch, start_decay_epoch, end_decay_epoch,
                         initial_lr_G_noisy_to_clean)
    adjust_learning_rate(optimizer_D_clean, epoch, start_decay_epoch, end_decay_epoch, initial_lr_D_clean)
    adjust_learning_rate(optimizer_D_noisy, epoch, start_decay_epoch, end_decay_epoch, initial_lr_D_noisy)
    if use_mbd:
        adjust_learning_rate(optimizer_D_clean_MBD, epoch, start_decay_epoch, end_decay_epoch, initial_lr_Ds_MBD)
        adjust_learning_rate(optimizer_D_noisy_MBD, epoch, start_decay_epoch, end_decay_epoch, initial_lr_Ds_MBD)

    for i, (clean_data, noisy_data) in tqdm(enumerate(zip(clean_loader, noisy_loader)),
                                            total=len(clean_loader), desc=f"Epoch {epoch + 1}/{num_epochs}"):

        real_clean = clean_data[0].to(device)
        real_noisy = noisy_data[0].to(device)

        if augment_training_data:
            real_clean = random_flip_tensors(real_clean)
            real_noisy = random_flip_tensors(real_noisy)

        if use_mbd:
            # for MBD discriminator
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

        # First, get the combined discriminator scores for the fake images
        # produced by the generators
        D_noisy_fake_noisy = D_noisy(fake_noisy)
        D_noisy_MBD_fake_noisy = D_noisy_MBD(fake_noisy)
        D_clean_fake_clean = D_clean(fake_clean)
        D_clean_MBD_fake_clean = D_clean_MBD(fake_clean)

        # Combine the PatchGAN and minibatch discriminator outputs
        combined_D_noisy_fake_noisy = D_noisy_fake_noisy * D_noisy_MBD_fake_noisy.unsqueeze(2).unsqueeze(
            3).expand_as(D_noisy_fake_noisy)
        combined_D_clean_fake_clean = D_clean_fake_clean * D_clean_MBD_fake_clean.unsqueeze(2).unsqueeze(
            3).expand_as(D_clean_fake_clean)

        # Now compute the adversarial losses using these combined scores
        # The generators are trained to make these combined scores close to the valid labels
        loss_GAN_clean_to_noisy_combined = adversarial_loss(combined_D_noisy_fake_noisy, valid)
        loss_GAN_noisy_to_clean_combined = adversarial_loss(combined_D_clean_fake_clean, valid)

        # Cycle loss
        recon_clean = G_noisy_to_clean(fake_noisy)
        recon_noisy = G_clean_to_noisy(fake_clean)
        loss_cycle_clean = cycle_loss(recon_clean, real_clean) * lambda_cycle
        loss_cycle_noisy = cycle_loss(recon_noisy, real_noisy) * lambda_cycle

        if use_identity_loss:
            # Identity loss
            identity_clean = G_noisy_to_clean(real_clean)
            identity_noisy = G_clean_to_noisy(recon_noisy)

            identity_loss_G_clean = identity_loss(real_clean, identity_clean) * lambda_cycle / 2
            identity_loss_G_noisy = identity_loss(recon_noisy, identity_noisy) * lambda_cycle / 2

            # Total generators losses
            loss_G_clean = (loss_cycle_clean + identity_loss_G_clean +
                            loss_GAN_clean_to_noisy_combined)
            loss_G_noisy = (loss_cycle_noisy + identity_loss_G_noisy +
                            loss_GAN_noisy_to_clean_combined)

            # Total generator loss
            loss_G = (loss_cycle_clean + loss_cycle_noisy + identity_loss_G_clean + identity_loss_G_noisy +
                      loss_GAN_clean_to_noisy_combined + loss_GAN_noisy_to_clean_combined)

        else:
            # Total generators losses
            loss_G_clean = (loss_cycle_clean +
                            loss_GAN_clean_to_noisy_combined)
            loss_G_noisy = (loss_cycle_noisy +
                            loss_GAN_noisy_to_clean_combined)

            # Total generator loss
            loss_G = (loss_cycle_clean + loss_cycle_noisy +
                      loss_GAN_clean_to_noisy_combined + loss_GAN_noisy_to_clean_combined)

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

        ##########
        D_clean_fake_clean = D_clean(fake_clean.detach())
        D_noisy_fake_noisy = D_noisy(fake_noisy.detach())

        D_clean_real_clean = D_clean(real_clean.detach())
        D_noisy_real_noisy = D_noisy(real_noisy.detach())

        D_clean_MBD_fake_clean = D_clean_MBD(fake_clean.detach())
        D_noisy_MBD_fake_noisy = D_noisy_MBD(fake_noisy.detach())

        D_clean_MBD_real_clean = D_clean_MBD(real_clean.detach())
        D_noisy_MBD_real_noisy = D_noisy_MBD(real_noisy.detach())

        # Real loss
        # Combine the PatchGAN and minibatch discriminator outputs
        combined_D_noisy_real_noisy = (D_noisy_real_noisy *
                                       D_noisy_MBD_real_noisy.unsqueeze(2).unsqueeze(3).expand_as(D_noisy_real_noisy))
        combined_D_clean_real_clean = (D_clean_real_clean *
                                       D_clean_MBD_real_clean.unsqueeze(2).unsqueeze(3).expand_as(D_clean_real_clean))

        # Fake loss
        # Combine the PatchGAN and minibatch discriminator outputs
        combined_D_noisy_fake_noisy = (D_noisy_fake_noisy *
                                       D_noisy_MBD_fake_noisy.unsqueeze(2).unsqueeze(3).expand_as(D_noisy_fake_noisy))
        combined_D_clean_fake_clean = (D_clean_fake_clean *
                                       D_clean_MBD_fake_clean.unsqueeze(2).unsqueeze(3).expand_as(D_clean_fake_clean))

        # Gradient penalty loss
        gradient_penalty_clean = compute_gradient_penalty(D_clean, real_clean.detach(), fake_clean.detach(),
                                                          center=l_constraint)
        gradient_penalty_noisy = compute_gradient_penalty(D_noisy, real_noisy.detach(), fake_noisy.detach(),
                                                          center=l_constraint)

        # Gradient penalty loss
        # gradient_penalty_clean_MBD = compute_gradient_penalty(D_clean_MBD, real_clean, fake_clean.detach())
        # gradient_penalty_noisy_MBD = compute_gradient_penalty(D_noisy_MBD, real_noisy, fake_noisy.detach())

        loss_D_clean = (adversarial_loss(combined_D_clean_real_clean, valid) +
                        adversarial_loss(combined_D_clean_fake_clean, fake)
                        + lambda_gp * gradient_penalty_clean)

        loss_D_noisy = (adversarial_loss(combined_D_noisy_real_noisy, valid) +
                        adversarial_loss(combined_D_noisy_fake_noisy, fake)
                        + lambda_gp * gradient_penalty_noisy)

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
            historical_params_G_clean_to_noisy[name].mul_(alpha_G).add_(param.data, alpha=1 - alpha_G)
            param.data.copy_(historical_params_G_clean_to_noisy[name])

        for name, param in G_noisy_to_clean.named_parameters():
            historical_params_G_noisy_to_clean[name].mul_(alpha_G).add_(param.data, alpha=1 - alpha_G)
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

        if (epoch % 1 == 0) and i == 0:  # Optionally display images every 5 epochs
            with torch.no_grad():
                # CLEAN TO NOISY
                generated_images_noisy = G_clean_to_noisy(real_clean)
                generated_generated_images = G_noisy_to_clean(generated_images_noisy)

                # calculate the score per image from the PatchGAN discriminator
                D1_score = D_noisy(generated_images_noisy).mean(dim=(-2, -1)) * D_noisy_MBD(generated_images_noisy)
                D2_score = D_clean(generated_generated_images).mean(dim=(-2, -1)) * D_clean_MBD(
                    generated_generated_images)

                show_images2(real_clean, generated_images_noisy, generated_generated_images,
                             D1_score, D2_score,
                             epoch, "clean", save_dir)

                # NOISY TO CLEAN
                generated_images = G_noisy_to_clean(real_noisy)
                generated_generated_images = G_clean_to_noisy(generated_images)

                # calculate the score per image from the PatchGAN discriminator
                D1_score = D_clean(generated_images).mean(dim=(-2, -1)) * D_clean_MBD(generated_images)
                D2_score = D_noisy(generated_generated_images).mean(dim=(-2, -1)) * D_noisy_MBD(
                    generated_generated_images)

                show_images2(real_noisy, generated_images, generated_generated_images,
                             D1_score, D2_score,
                             epoch, "noisy", save_dir)
        # ------------------------
        #  Calculate the FID Score of the batch
        # ------------------------

        with torch.no_grad():
            real_batch_features = fid_calculator.extract_features(real_noisy.detach().to('cpu'))
            fake_batch_features = fid_calculator.extract_features(generated_images_noisy.detach().to('cpu'))
            real_features.append(real_batch_features)
            fake_features.append(fake_batch_features)

    # At the end of the epoch, concatenate all batch features
    real_features_epoch = np.concatenate(real_features, axis=0)
    fake_features_epoch = np.concatenate(fake_features, axis=0)

    # Compute FID score using the aggregated features
    mu_real, sigma_real = fid_calculator.calculate_statistics(real_features_epoch)
    mu_fake, sigma_fake = fid_calculator.calculate_statistics(fake_features_epoch)
    fid_score = fid_calculator.calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)

    fid_score_list.append(fid_score)
    fid_epochs.append(epoch)
    print(f"Epoch {epoch + 1}, FID score: {fid_score}")

    # if epoch % 5 == 0:
    #     # Calculate FID Score
    #     fid_score = fid_calculator.get_fid_score(real_noisy, generated_images_noisy)
    #     fid_score_list.append(fid_score)
    #     fid_epochs.append(epoch)
    #     print(f"FID score: {fid_score}")

    if (epoch % 50 == 0) and (epoch != 0):
        # Save model weights
        torch.save(G_clean_to_noisy.state_dict(), save_dir + '/G_clean_to_noisy_weights_' + str(epoch) + '.pth')
        torch.save(G_noisy_to_clean.state_dict(), save_dir + '/G_noisy_to_clean_weights_' + str(epoch) + '.pth')
        torch.save(D_clean.state_dict(), save_dir + '/D_clean_weights_' + str(epoch) + '.pth')
        torch.save(D_noisy.state_dict(), save_dir + '/D_noisy_weights_' + str(epoch) + '.pth')
        if use_mbd:
            torch.save(D_clean_MBD.state_dict(), save_dir + '/D_clean_MBD_weights_' + str(epoch) + '.pth')
            torch.save(D_noisy_MBD.state_dict(), save_dir + '/D_noisy_MBD_weights_' + str(epoch) + '.pth')

        # Save entire models
        torch.save(G_clean_to_noisy, save_dir + '/G_clean_to_noisy_' + str(epoch) + '.pth')
        torch.save(G_noisy_to_clean, save_dir + '/G_noisy_to_clean_' + str(epoch) + '.pth')
        torch.save(D_clean, save_dir + '/D_clean_' + str(epoch) + '.pth')
        torch.save(D_noisy, save_dir + '/D_noisy_' + str(epoch) + '.pth')
        if use_mbd:
            torch.save(D_clean_MBD, save_dir + '/D_clean_MBD_' + str(epoch) + '.pth')
            torch.save(D_noisy_MBD, save_dir + '/D_noisy_MBD_' + str(epoch) + '.pth')

    losses_G_clean_to_noisy.append(loss_GAN_clean_to_noisy_combined.item())
    losses_G_noisy_to_clean.append(loss_GAN_noisy_to_clean_combined.item())
    losses_D_clean.append(loss_D_clean.item())
    losses_D_noisy.append(loss_D_noisy.item())

    # losses_cycle_clean.append(loss_cycle_clean.item())
    # losses_cycle_noisy.append(loss_cycle_noisy.item())
    #
    # losses_identity_clean.append(identity_loss_G_clean.item())
    # losses_identity_noisy.append(identity_loss_G_noisy.item())
    #
    # losses_D_clean_patch.append((loss_real_clean.item() + loss_fake_clean.item()) / 2)
    # losses_D_noisy_patch.append((loss_real_noisy.item() + loss_fake_noisy.item()) / 2)
    #
    # losses_D_clean_MBD.append((loss_real_clean_MBD.item() + loss_fake_clean_MBD.item()) / 2)
    # losses_D_noisy_MBD.append((loss_real_noisy_MBD.item() + loss_fake_noisy_MBD.item()) / 2)

    print(f"Epoch [{epoch + 1}/{num_epochs}] -"
          f" Loss D_clean: {loss_D_clean.item():.5f},"
          f" Loss D_noisy: {loss_D_noisy.item():.5f},"
          f" Loss G_Clean_to_Noisy: {loss_G_clean.item():.5f},"
          f" Loss G_Noisy_to_Clean: {loss_G_noisy.item():.5f}")


    print(f"Epoch [{epoch + 1}/{num_epochs}] -"
          f" gp_clean: {gradient_penalty_clean.item():.5f},"
          f" gp_noisy: {gradient_penalty_noisy.item():.5f},"
          )

    plot_losses(losses_G_clean_to_noisy, losses_G_noisy_to_clean,
                losses_D_clean, losses_D_noisy,
                save_dir)

    if ree:
        plot_D_losses2(losses_D_clean_patch, losses_D_noisy_patch,
                           losses_D_clean_MBD, losses_D_noisy_MBD,
                           losses_cycle_clean, losses_cycle_noisy,
                           losses_identity_clean, losses_identity_noisy,
                           save_dir)

    plot_fid_scores(fid_epochs, fid_score_list, save_dir)

# Save model weights
torch.save(G_clean_to_noisy.state_dict(), save_dir + '/G_clean_to_noisy_weights.pth')
torch.save(G_noisy_to_clean.state_dict(), save_dir + '/G_noisy_to_clean_weights.pth')
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

ree = False

if ree:
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

    # Gradient penalty loss
    gradient_penalty_clean = compute_gradient_penalty(D_clean, real_clean, fake_clean.detach())
    gradient_penalty_noisy = compute_gradient_penalty(D_noisy, real_noisy, fake_noisy.detach())

    # Gradient penalty loss
    gradient_penalty_clean_MBD = compute_gradient_penalty(D_clean_MBD, real_clean, fake_clean.detach())
    gradient_penalty_noisy_MBD = compute_gradient_penalty(D_noisy_MBD, real_noisy, fake_noisy.detach())

    loss_D_clean = ((loss_real_clean + loss_fake_clean + lambda_gp * gradient_penalty_clean) *
                    (loss_real_clean_MBD + loss_fake_clean_MBD))

    loss_D_noisy = ((loss_real_noisy + loss_fake_noisy + lambda_gp * gradient_penalty_noisy)
                    * (loss_real_noisy_MBD + loss_fake_noisy_MBD))

    loss_D_clean.backward()
    loss_D_noisy.backward()

    optimizer_D_clean.step()
    optimizer_D_noisy.step()

    optimizer_D_clean_MBD.step()
    optimizer_D_noisy_MBD.step()
