# from .gan_loss import gan_loss
# from .adversarial_loss import adversarial_loss
# from .perturbation_loss import perturbation_loss
import torch
import torch.nn.functional as F
import torch 
__all__ = [
    'gan_loss',
    'adversarial_loss',
    'perturbation_loss'
]

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def gan_loss(D, real_images, fake_images):
    real_loss = torch.mean(torch.log(D(real_images) + 1e-10))
    fake_loss = torch.mean(torch.log(1 - D(fake_images) + 1e-10))
    return -(real_loss + fake_loss)


def adversarial_loss(vae, x_prime, watermark):
    latent_watermark = vae.encode(watermark)
    latent_adversarial = vae.encode(x_prime)
    return (l2norm(latent_adversarial - latent_watermark, dim=1)**2).mean()
    # return F.mse_loss(latent_adversarial, latent_watermark) 


def perturbation_loss(perturbation, watermark, c=0.1, watermark_region=4):
    weighted_perturbation = torch.matmul(perturbation, (1 + watermark * watermark_region))
    # l2_norm = torch.norm(weighted_perturbation, p=2)
    l2_norm = l2_norm(weighted_perturbation, dim=1)
    mask = ((l2_norm - c)>0) + 0 
    threshold_loss = (l2_norm * mask).mean()
    # threshold_loss = torch.max(torch.zeros_like(l2_norm), l2_norm - c)
    
    return threshold_loss # + small_perturbation_penalty

