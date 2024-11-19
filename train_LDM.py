import torch
import torch.optim as optim

from models import LDM, Generator, Discriminator
from data_loader import get_dataloader

def train_adversarial(generator, discriminator, ldm, dataloader, num_epochs, device='cuda', checkpoint=None, load_checkpoint=False):
    """
    Train function focusing on optimizing adversarial loss with pre-trained Generator
    """
    # Load pretrained models
    start_epoch = 0
    if load_checkpoint:
        checkpoint = torch.load(checkpoint)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded pretrained models")
    for param in ldm.vae.parameters():
        param.requires_grad = False
    
    generator.to(device)
    discriminator.to(device)
    
    # Freeze discriminator
    discriminator.eval()
    for param in discriminator.parameters():
        param.requires_grad = False

    # Only optimize generator
    optimizer_G = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))

    # Hyperparameters
    # lambda_gan = 1.0    # Weight for GAN loss
    lambda_adv = 1.0     # Weight for adversarial loss
    lambda_pert = 10    # Small weight for perturbation loss to maintain stability
    c = 10/255          # Bound for perturbation

    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_adv_loss = 0
        total_batches = 0
        
        for i, (images, watermarks) in enumerate(dataloader):
            batch_size = images.size(0)
            images = images.to(device)
            watermarks = watermarks.to(device)

            # Train Generator for adversarial attack
            optimizer_G.zero_grad()

            # Generate perturbation and get adversarial examples
            perturbation = generator(images, watermarks)
            x_prime = images + perturbation
            x_prime = torch.clamp(x_prime, -1, 1)  # Ensure valid image range

            # Adversarial loss against DMs
            g_loss_adv = compute_adv_loss(x_prime, watermarks, ldm)

            # Small perturbation loss to maintain stability
            l2_norm = torch.norm(perturbation.view(batch_size, -1), p=2, dim=1)
            g_loss_pert = torch.mean(torch.max(torch.zeros_like(l2_norm), l2_norm - c))

            # Combined loss (focus on adversarial)
            g_loss = lambda_adv * g_loss_adv + lambda_pert * g_loss_pert
            g_loss.backward()
            optimizer_G.step()

            # Logging
            total_adv_loss += g_loss_adv.item()
            total_batches += 1

            if i % 10 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[Adv loss: {g_loss_adv.item():.4f}] "
                      f"[Pert loss: {g_loss_pert.item():.4f}] "
                      f"[Total loss: {g_loss.item():.4f}]")

        # Print epoch statistics
        avg_adv_loss = total_adv_loss / total_batches
        # print(f"Epoch {epoch+1}/{num_epochs} - Average Adversarial Loss: {avg_adv_loss:.4f}")

        # Save generator periodically
        if (epoch+1) % 5 == 0:
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'epoch': epoch+1,
                'adv_loss': avg_adv_loss
            }, f'checkpoints/adv/generator_adv_epoch_{epoch+1}.pth')

def compute_adv_loss(x_prime, watermark, ldm):
    """
    Compute adversarial loss in latent space between x' and m
    Args:
        x_prime: Adversarial image from generator (batch_size, channels, H, W)
        watermark: Watermark image (batch_size, channels, H, W)
        ldm: Latent Diffusion Model (VAE encoder) used to get latent representations
    Returns:
        L_adv: Adversarial loss between latent representations
    """
    eps_x_prime = ldm.get_latent(x_prime)
    eps_m = ldm.get_latent(watermark) 
    return torch.mean(torch.norm(eps_x_prime - eps_m, p=2, dim=1))

if __name__ == '__main__':
    # Load pretrained models
    generator = Generator(input_channels=6)
    discriminator = Discriminator(input_channels=3)
    ldm = LDM()

    data_dir = 'data/val/'
    dataloader = get_dataloader(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train adversarial
    num_epochs = 100
    train_adversarial(
        generator, 
        discriminator,
        ldm,
        dataloader,
        num_epochs,
        device,
        checkpoint='checkpoints/checkpoint_epoch_20.pth',
        load_checkpoint=True
    )