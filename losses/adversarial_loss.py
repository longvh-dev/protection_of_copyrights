import torch.nn.functional as F

def adversarial_loss(vae, x_prime, watermark):
    latent_watermark = vae.encode(watermark)
    latent_adversarial = vae.encode(x_prime)
    return F.mse_loss(latent_adversarial, latent_watermark) 