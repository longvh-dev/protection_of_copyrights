from diffusers import AutoencoderKL
import torch

class VAEWrapper:
    def __init__(self, pretrained_path="runwayml/stable-diffusion-v1-5"):
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_path,
            torch_dtype=torch.float32,
            subfolder="vae",
        )
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def encode(self, x):
        x = x.clamp(0, 1)
        x = 2 * x - 1  # Scale to [-1, 1]
        # with torch.no_grad():
        latent = self.vae.encode(x).latent_dist.mean
        latent = latent * self.vae.config.scaling_factor
        return latent
    
    def decode(self, z):
        with torch.no_grad():
            z = z / self.vae.config.scaling_factor
            x = self.vae.decode(z)
            x = (x + 1) / 2
    
    def to(self, device):
        self.vae = self.vae.to(device)
        return self