import torch
from transformers import AutoencoderKL

class LDM:
    def __init__(self, device='cuda'):
        # Load pretrained VAE
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
        self.vae.eval()  # Set to evaluation mode
        
    def get_latent(self, x):
        # Get latent representation without gradient computation
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
            return latent