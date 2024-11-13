import torch

def perturbation_loss(perturbation, watermark, c=0.1, watermark_region=4):
    weighted_perturbation = perturbation * (1 + watermark * watermark_region)
    l2_norm = torch.norm(weighted_perturbation, p=2)
    return torch.mean(torch.max(torch.zeros_like(l2_norm), l2_norm - c))
