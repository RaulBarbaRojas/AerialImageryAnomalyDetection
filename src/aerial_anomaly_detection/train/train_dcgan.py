"""
Module to train the DCGAN model on the LandCover.ai dataset.

Radford, A. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks.\
arXiv preprint arXiv:1511.06434.

NOTE: This custom module is used due to the differences that DCGAN has with respect to "normal" architectures.
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
from aerial_anomaly_detection.models.utils import Decoder as Generator
from aerial_anomaly_detection.models.utils import Discriminator


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

    # Step 2: Preparing DCGAN model
    latent_dimension = run_ctx.params.get('latent_dimension', 1000)
    img_width = run_ctx.params.get('img_width', 32)
    img_height = run_ctx.params.get('img_height', 32)

    pretrained_weight_folder = run_ctx.params.get('pretrained_weights', None)
    generator = Generator(latent_dimension, img_width, img_height).to(device)
    discriminator = Discriminator(img_width, img_height).to(device)

    if pretrained_weight_folder is not None:
        pretrained_weight_folder = Path(pretrained_weight_folder)
        generator_keys = generator.load_state_dict(torch.load(pretrained_weight_folder / 'generator.pth', weights_only = True))
        discriminator_keys = discriminator.load_state_dict(torch.load(pretrained_weight_folder / 'discriminator.pth', weights_only = True))
        print('Generator weights loaded: ', generator_keys)
        print('Discriminator weights loaded: ', discriminator_keys)
    else:
        generator.apply(weight_initialization)
        discriminator.apply(weight_initialization)

    # Step 3: Running training
    n_epochs = run_ctx.params.get('n_epochs', 20)
    start_epoch = run_ctx.params.get('start_epoch', 0)
    epochs_per_save = run_ctx.params.get('epochs_per_save', 5)
    num_evaluation_images = run_ctx.params.get('num_evaluation_images', 64)
    learning_rate = run_ctx.params.get('learning_rate', 0.0001)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr = learning_rate, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr = learning_rate, betas=(0.5, 0.999))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    fixed_noise_vector = torch.randn(num_evaluation_images, latent_dimension, device=device)
    for epoch in tqdm(range(start_epoch, start_epoch + n_epochs),
                      desc = f'Training DCGAN during {n_epochs} epochs',
                      unit = 'epoch',
                      file = sys.stdout,
                      dynamic_ncols = True):

        # Step 1: Training DCGAN
        generator.train()
        discriminator.train()

        for _, X, y in tqdm(train_dataloader,
                            desc = 'Processing epoch training batches',
                            unit = 'batch',
                            dynamic_ncols = True,
                            file = sys.stdout):
            X, y = X.to(device), y.to(device)

            # Step 1.1: Training the discriminator
            discriminator.zero_grad()
            labels = torch.full((X.shape[0], ), 1.0, dtype = torch.float, device = device)
            y_pred = discriminator(X).view(-1)
            discriminator_batch_real_train_loss = loss_fn(y_pred, labels)
            discriminator_batch_real_train_loss.backward()

            sample_noise_vector = torch.randn((X.shape[0], latent_dimension), device = device)
            fake_data = generator(sample_noise_vector)
            fake_labels = torch.full((X.shape[0], ), 0.0, dtype = torch.float, device = device)
            y_pred = discriminator(fake_data.detach()).view(-1)
            discriminator_batch_fake_train_loss = loss_fn(y_pred, fake_labels)
            discriminator_batch_fake_train_loss.backward()

            discriminator_batch_loss = discriminator_batch_real_train_loss + discriminator_batch_fake_train_loss
            discriminator_optimizer.step()

            # Step 1.2: Training the generator
            generator.zero_grad()
            labels = torch.full((X.shape[0], ), 1.0, dtype = torch.float, device = device)
            y_pred = discriminator(fake_data).view(-1)
            generator_batch_loss = loss_fn(y_pred, labels)
            generator_batch_loss.backward()
            generator_optimizer.step()

        # Step 2: Saving current DCGAN and its performance over fixed noise vector
        if epoch % epochs_per_save == 0:
            generator.eval()
            discriminator.eval()
            (epoch_out_folder := out_folder / f'epoch_{epoch}').mkdir(exist_ok = True, parents = True)
            torch.save(generator.state_dict(), epoch_out_folder / 'generator.pth')
            torch.save(discriminator.state_dict(), epoch_out_folder / 'discriminator.pth')

        print(f'Epoch {epoch} | Discriminator loss: {discriminator_batch_loss.item():.6f} | '
              f'Generator loss: {generator_batch_loss.item():.6f}')

    # Step 3: Storing final results
    generator.eval()
    discriminator.eval()
    (epoch_out_folder := out_folder / f'epoch_{epoch}').mkdir(exist_ok = True, parents = True)
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
