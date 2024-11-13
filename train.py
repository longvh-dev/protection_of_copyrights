import torch
from torch.utils.data import DataLoader
from datetime import datetime
import os

from models import Generator, Discriminator, VAEWrapper
from losses import gan_loss, adversarial_loss, perturbation_loss
from config import TrainingConfig, ModelConfig
from data_loader import create_dataloader

def train_step(G, D, vae, optimizer_G, optimizer_D, real_images, watermark, config):
    device = real_images.device

    # Train Discriminator
    optimizer_D.zero_grad()
    perturbation = G(real_images, watermark).detach()
    fake_images = real_images + perturbation
    d_loss = gan_loss(D, real_images, fake_images)
    d_loss.backward()
    optimizer_D.step()
    
    # Train Generator
    optimizer_G.zero_grad()
    perturbation = G(real_images, watermark)
    fake_images = real_images + perturbation
    
    g_loss = adversarial_loss(vae, fake_images, watermark) + \
             config.alpha * gan_loss(D, real_images, fake_images) + \
             config.beta * perturbation_loss(perturbation, watermark, config.c, config.watermark_region)
    
    g_loss.backward()
    optimizer_G.step()
    
    return d_loss.item(), g_loss.item(), fake_images

def main():
    # Load configs
    train_config = TrainingConfig()
    model_config = ModelConfig()

    # Load dataset
    image_dir = "data/wikiart"
    classes_csv = "data/wikiart/train_classes.csv"
    
    # Create dataloader
    dataloader, metadata = create_dataloader(
        image_dir=image_dir,
        classes_csv=classes_csv,
        batch_size=train_config.batch_size,
    )
    
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(model_config.input_channels).to(device)
    D = Discriminator(model_config.input_channels).to(device)
    vae = VAEWrapper(model_config.vae_path).to(device)
    
    # Setup optimizers
    optimizer_G = torch.optim.Adam(
        G.parameters(), 
        lr=train_config.lr, 
        betas=(train_config.beta1, train_config.beta2)
    )
    optimizer_D = torch.optim.Adam(
        D.parameters(),
        lr=train_config.lr,
        betas=(train_config.beta1, train_config.beta2)
    )
    
    # Training loop
    start_epoch = 0
    if train_config.checkpoint:
        checkpoint = torch.load(train_config.checkpoint)
        start_epoch = checkpoint['epoch']
        G.load_state_dict(checkpoint['generator_state_dict'])
        D.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{train_config.save_dir}/{timestamp}"
    # make save_dir

    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(start_epoch, train_config.num_epochs+1):
        for batch_idx, (real_images, watermark, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            watermark = watermark.to(device)
            
            d_loss, g_loss, fake_images = train_step(
                G, D, vae, optimizer_G, optimizer_D,
                real_images, watermark, train_config
            )
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{train_config.num_epochs}] "
                      f"Batch [{batch_idx}] D_loss: {d_loss:.4f} "
                      f"G_loss: {g_loss:.4f}")
        
        # Save models every 10 epochs
        if (epoch) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': G.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, f"{save_dir}/checkpoint_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()