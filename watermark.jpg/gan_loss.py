import torch

def gan_loss(D, real_images, fake_images):
    real_loss = torch.mean(torch.log(D(real_images) + 1e-10))
    fake_loss = torch.mean(torch.log(1 - D(fake_images) + 1e-10))
    return -(real_loss + fake_loss)