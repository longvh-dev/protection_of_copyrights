import torch

def perturbation_loss(perturbation, watermark, c=0.1, watermark_region=4):
    weighted_perturbation = torch.matmul(perturbation, (1 + watermark * watermark_region))
    l2_norm = torch.norm(weighted_perturbation, p=2)
    
    # size_penalty = torch.exp(-l2_norm) if l2_norm < c else 0
    
    threshold_loss = torch.max(torch.zeros_like(l2_norm), l2_norm - c)
    
    # small_perturbation_penalty = size_penalty * 10

    return threshold_loss # + small_perturbation_penalty
