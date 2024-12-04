import torch.nn.functional as F
import torch 
def adversarial_loss(vae, x_prime, watermark):
    latent_watermark = vae.encode(watermark)
    latent_adversarial = vae.encode(x_prime)
    return torch(latent_adversarial - latent_adversarial)
    # return F.mse_loss(latent_adversarial, latent_watermark) 