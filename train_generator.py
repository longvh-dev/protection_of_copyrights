
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

from models.generator import Generator
from models.discriminator import Discriminator
from data_loader import get_dataloader
from utils import GANLoss

# Training function
def train(generator, discriminator, dataloader, num_epochs, device='cuda', save_model_dir='checkpoints', checkpoint=None, load_checkpoint=False):
    start_epoch = 0
    if load_checkpoint:
        checkpoint = torch.load(checkpoint)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}")

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    
    generator.to(device)
    discriminator.to(device)
    # target_model.to(device)
    # target_model.eval()  # Freeze target model

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    gan_loss = GANLoss().to(device)

    # Hyperparameters
    lambda_gan = 1.0    # Weight for GAN loss
    lambda_pert = 10.0  # Weight for perturbation loss
    w = 4.0            # Weight for watermark region
    c = 10/255        # Bound for perturbation

    for epoch in range(start_epoch, start_epoch + num_epochs):
        for i, (images, watermarks) in enumerate(dataloader):
            batch_size = images.size(0)
            images = images.to(device)
            watermarks = watermarks.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Real images
            d_real = discriminator(images)
            d_real_loss = gan_loss(d_real, True)

            # Generate perturbation and get adversarial examples
            perturbation = generator(images, watermarks)
            x_prime = images + perturbation
            x_prime = torch.clamp(x_prime, -1, 1)  # Ensure valid image range

            # Fake images
            d_fake = discriminator(x_prime.detach())
            d_fake_loss = gan_loss(d_fake, False)

            # Combined discriminator loss
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            # GAN loss
            d_fake = discriminator(x_prime)
            g_loss_gan = gan_loss(d_fake, True)

            # Perturbation loss with weighted watermark region
            weighted_perturbation = perturbation * (1 + w * watermarks)
            l2_norm = torch.norm(weighted_perturbation.view(batch_size, -1), p=2, dim=1)
            g_loss_pert = torch.mean(torch.max(torch.zeros_like(l2_norm), l2_norm - c))

            # Combined generator loss
            g_loss = lambda_gan * g_loss_gan + lambda_pert * g_loss_pert
            g_loss.backward()
            optimizer_G.step()

            if i % 10 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                      f"[Pert loss: {g_loss_pert.item():.4f}]")

        # Save models periodically
        if (epoch+1) % 5 == 0:
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'epoch': epoch+1,
            }, f'{save_model_dir}/checkpoint_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    # Initialize models
    generator = Generator(input_channels=6)
    discriminator = Discriminator(input_channels=3)

    data_dir = 'data/train/'
    dataloader = get_dataloader(data_dir)

    save_model_dir = 'checkpoints/train_10_samples/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Train
    num_epochs = 20
    train(generator, discriminator, dataloader, num_epochs, device, save_model_dir, checkpoint='checkpoint_epoch_5.pth', load_checkpoint=False)