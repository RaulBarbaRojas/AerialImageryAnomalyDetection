"""
Module to train the BiGAN model on the LandCover.ai dataset.

Donahue, J., Krähenbühl, P., & Darrell, T. (2016). Adversarial feature learning. arXiv preprint arXiv:1605.09782.

NOTE: This custom module is used due to the differences that BiGAN has with respect to "normal" architectures.
"""


import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from afml.context import run_ctx
from tqdm.auto import tqdm

from aerial_anomaly_detection.datasets import DataLoader, Dataset
from aerial_anomaly_detection.models.bigan import BiGANDiscriminator
from aerial_anomaly_detection.models.utils import Encoder, Decoder as Generator


def weight_initialization(torch_module : torch.nn.Module) -> None:
    """
    Function to run weight initialization as described in the original paper over a given layer/torch module.

    Args:
        torch_module (Module): the module/layer whose weights will be initialized.
    """
    classname = torch_module.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(torch_module.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(torch_module.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(torch_module.bias.data, 0)


if __name__ == '__main__':

    # Step 1: Preparing training data
    (out_folder := Path(run_ctx.params.out_folder)).mkdir(exist_ok = True, parents = True)
    batch_size = run_ctx.params.get('batch_size', 256)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_dataset = Dataset.load(run_ctx.dataset.params.processed_folder, partition = 'train')
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = os.cpu_count())

    # Step 2: Preparing BiGAN model
    latent_dimension = run_ctx.params.get('latent_dimension', 1000)
    img_width = run_ctx.params.get('img_width', 32)
    img_height = run_ctx.params.get('img_height', 32)

    pretrained_weight_folder = run_ctx.params.get('pretrained_weights', None)
    encoder = Encoder(latent_dimension, img_width, img_height).to(device)
    generator = Generator(latent_dimension, img_width, img_height).to(device)
    discriminator = BiGANDiscriminator(latent_dimension, img_width, img_height).to(device)

    if pretrained_weight_folder is not None:
        pretrained_weight_folder = Path(pretrained_weight_folder)
        encoder_keys = encoder.load_state_dict(torch.load(pretrained_weight_folder / 'encoder.pth', weights_only = True))
        generator_keys = generator.load_state_dict(torch.load(pretrained_weight_folder / 'generator.pth', weights_only = True))
        discriminator_keys = discriminator.load_state_dict(torch.load(pretrained_weight_folder / 'discriminator.pth', weights_only = True))
        print('Encoder weights loaded: ', encoder_keys)
        print('Generator weights loaded: ', generator_keys)
        print('Discriminator weights loaded: ', discriminator_keys)
    else:
        encoder.apply(weight_initialization)
        generator.apply(weight_initialization)
        discriminator.apply(weight_initialization)

    # Step 3: Running training
    n_epochs = run_ctx.params.get('n_epochs', 20)
    start_epoch = run_ctx.params.get('start_epoch', 0)
    epochs_per_save = run_ctx.params.get('epochs_per_save', 5)
    num_evaluation_images = run_ctx.params.get('num_evaluation_images', 64)
    learning_rate = run_ctx.params.get('learning_rate', 0.0001)
    encoder_generator_optimizer = torch.optim.Adam(list(generator.parameters()) + list(encoder.parameters()),
                                                   lr = learning_rate, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr = learning_rate, betas=(0.5, 0.999))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    fixed_noise_vector = torch.randn(num_evaluation_images, latent_dimension, device=device)
    for epoch in tqdm(range(start_epoch, start_epoch + n_epochs),
                      desc = f'Training BiGAN during {n_epochs} epochs',
                      unit = 'epoch',
                      file = sys.stdout,
                      dynamic_ncols = True):

        encoder.train()
        generator.train()
        discriminator.train()

        encoder_generator_epoch_loss = 0.0
        discriminator_epoch_loss = 0.0

        for _, X, y in tqdm(train_dataloader,
                            desc = 'Processing epoch training batches',
                            unit = 'batch',
                            dynamic_ncols = True,
                            file = sys.stdout):
            X, y = X.to(device), y.to(device)
            one_labels = torch.full((X.shape[0], ), 1.0, dtype = torch.float, device = device)
            zero_labels = torch.full((X.shape[0], ), 0.0, dtype = torch.float, device = device)
            noise_vector = 2 * torch.rand((X.shape[0], latent_dimension), device = device) - 1

            # Step 3.1: Encoder and Generator forward pass
            y_pred_generator = generator(noise_vector)
            y_pred_encoder = encoder(X)

            # Step 3.2: Discriminator forward pass of true and fake data
            y_pred_discriminator_gen = discriminator(y_pred_generator.detach(), noise_vector)[0].reshape(-1)
            y_pred_discriminator_enc = discriminator(X, y_pred_encoder.detach())[0].reshape(-1)

            # Step 3.3: Loss calculation
            discriminator_loss = loss_fn(torch.cat([y_pred_discriminator_enc, y_pred_discriminator_gen]),
                                         torch.cat([one_labels, zero_labels]))
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            discriminator_epoch_loss += discriminator_loss.item()

            # Step 3.5: Encoder/Generator training
            y_pred_discriminator_gen = discriminator(y_pred_generator, noise_vector)[0].reshape(-1)
            y_pred_discriminator_enc = discriminator(X, y_pred_encoder)[0].reshape(-1)

            encoder_generator_loss = loss_fn(torch.cat([y_pred_discriminator_gen, y_pred_discriminator_enc]),
                                             torch.cat([one_labels, zero_labels]))
            encoder_generator_optimizer.zero_grad()
            encoder_generator_loss.backward()
            encoder_generator_optimizer.step()

            encoder_generator_epoch_loss += encoder_generator_loss.item()

        # Step 3.6: Saving current BiGAN and its performance over fixed noise vector
        encoder_generator_epoch_loss /= len(train_dataloader)
        discriminator_epoch_loss /= len(train_dataloader)

        if epoch % epochs_per_save == 0:
            encoder.eval()
            generator.eval()
            discriminator.eval()
            (epoch_out_folder := out_folder / f'epoch_{epoch}').mkdir(exist_ok = True, parents = True)
            torch.save(encoder.state_dict(), epoch_out_folder / 'encoder.pth')
            torch.save(generator.state_dict(), epoch_out_folder / 'generator.pth')
            torch.save(discriminator.state_dict(), epoch_out_folder / 'discriminator.pth')

        print(f'Epoch {epoch} | Discriminator loss: {discriminator_epoch_loss:.6f} | '
              f'Encoder/Generator loss: {encoder_generator_epoch_loss:.6f}')

    # Step 4: Storing final results
    encoder.eval()
    generator.eval()
    discriminator.eval()
    (epoch_out_folder := out_folder / f'epoch_{epoch}').mkdir(exist_ok = True, parents = True)
    torch.save(encoder.state_dict(), epoch_out_folder / 'encoder.pth')
    torch.save(generator.state_dict(), epoch_out_folder / 'generator.pth')
    torch.save(discriminator.state_dict(), epoch_out_folder / 'discriminator.pth')

    with torch.inference_mode():
        generated_images = generator(fixed_noise_vector).cpu().numpy()
        generated_images = (((generated_images + 1) / 2) * 255).astype(np.uint8)

        num_plots_per_row = int(math.sqrt(fixed_noise_vector.shape[0]))
        _, axs = plt.subplots(nrows = num_plots_per_row,
                                ncols = num_plots_per_row,
                                figsize = (20, 20))
        for idx_image in range(fixed_noise_vector.shape[0]):
            ax = axs[idx_image // num_plots_per_row, idx_image % num_plots_per_row]
            ax.imshow(generated_images[idx_image, ...].squeeze().transpose(1, 2, 0))
            ax.axis('off')

        plt.margins(0, 0)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.savefig(epoch_out_folder / 'sample_generated_images.png')
        plt.close()
