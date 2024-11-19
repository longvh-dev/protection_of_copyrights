from diffusers import AutoencoderKL
import torch

class VAEWrapper:
    def __init__(self, pretrained_path="stabilityai/sd-vae-ft-mse"):
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_path,
            torch_dtype=torch.float32
        )
        self.vae.eval()
    
    def encode(self, x):
        x = 2 * x - 1  # Scale to [-1, 1]
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent
    
    def to(self, device):
        self.vae = self.vae.to(device)
        return self