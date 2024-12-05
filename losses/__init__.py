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

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def gan_loss(D, real_images, fake_images):
    real_pred = D(real_images)
    fake_pred = D(fake_images)
    
    real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
    fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
    
    return real_loss + fake_loss


def adversarial_loss(vae, x_prime, watermark):
    latent_watermark = l2norm(vae.encode(watermark))
    latent_adversarial = l2norm(vae.encode(x_prime))
    return torch.norm(latent_watermark-latent_adversarial, p=2, dim=1).mean()
    # return F.mse_loss(latent_adversarial, latent_watermark) 


def perturbation_loss(perturbation, watermark, c=0.1, watermark_region=4):
    weighted_perturbation = torch.matmul(perturbation, (1 + watermark * watermark_region))
    # l2_norm = torch.norm(weighted_perturbation, p=2)
    weighted_perturbation = l2norm(weighted_perturbation, dim=1)
    mask = ((weighted_perturbation - c)>0) + 0 
    threshold_loss = (weighted_perturbation * mask).mean()
    # threshold_loss = torch.max(torch.zeros_like(l2_norm), l2_norm - c)
    
    return threshold_loss # + small_perturbation_penalty

