import sys
import os
# sys.path.insert(0, os.path.join(os.path.__file__, ".."))

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
import os
import argparse
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms, models


from data_loader import create_watermark
from models import Generator, Discriminator, VAEWrapper
from losses import gan_loss, adversarial_loss, perturbation_loss
# from config import TrainingConfig, ModelConfig
from data_loader import create_dataloader
from tools.evaluate import evaluate_adversarial_quality


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--name", type=str, default="trial")
    args.add_argument("--checkpoint", type=str, default=None)
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--alpha", type=float, default=1)
    args.add_argument("--beta", type=float, default=10)
    args.add_argument("--c", type=float, default=10/255)
    args.add_argument("--watermark_region", type=float, default=4.0)
    args.add_argument("--input_channels", type=int, default=3)
    args.add_argument("--vae_path", type=str,
                      default="runwayml/stable-diffusion-v1-5")
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--save_dir", type=str, default="checkpoints")
    args.add_argument("--beta1", type=float, default=0.5)
    args.add_argument("--beta2", type=float, default=0.999)
    args.add_argument("--device", type=str, default="cuda")

    args.add_argument("--train_dir", type=str, default="data/wikiart")
    args.add_argument("--train_classes", type=str,
                      default="data/wikiart/train_classes.csv")
    args.add_argument("--eval_dir", type=str, default="data/wikiart")
    args.add_argument("--eval_classes", type=str,
                      default="data/wikiart/eval_classes.csv")

    return args.parse_args()

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif hasattr(m, 'weight') and classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


def train_step(G, D, vae, optimizer_G, optimizer_D, real_images, watermark, config):
    device = real_images.device
    current_batch_size = real_images.size(0)

    # Train Discriminator
    for _ in range(1):
        perturbation = G(real_images, watermark)
        adv_images = torch.clamp(perturbation, -0.3, 0.3) + real_images
        adv_images = torch.clamp(adv_images, 0, 1)
        
        optimizer_D.zero_grad()
        
        pred_real = D(real_images)
        loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=device))
        loss_D_real.backward()
        
        pred_fake = D(adv_images.detach())
        loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=device))
        loss_D_fake.backward()
        
        loss_D_GAN = loss_D_real + loss_D_fake
        optimizer_D.step()

    # Train Generator
    for _ in range(1):
        optimizer_G.zero_grad()
        pred_fake = D(adv_images)
        loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=device))
        loss_G_fake.backward(retain_graph=True)

        loss_adv = adversarial_loss(vae, adv_images, watermark)
        loss_perturbation = perturbation_loss(perturbation, watermark, config.c, config.watermark_region)
        # perturbation_loss_ = 0
        g_loss = loss_adv  + config.beta * loss_perturbation

        g_loss.backward()
        optimizer_G.step()
        
        g_loss_sum = config.alpha * loss_G_fake + loss_adv + config.beta * loss_perturbation

    return loss_D_GAN.item(), loss_G_fake.item(), loss_adv.item(), loss_perturbation.item(), g_loss_sum.item()


def main(args, pipe):
    test_image = Image.open('data/wikiart/Early_Renaissance/andrea-del-castagno_dante-alighieri.jpg').convert("RGB")
    test_image_size = test_image.size[::-1]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{args.save_dir}/{args.name}/{timestamp}"
    # make save_dir

    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    train_image_dir = args.train_dir
    train_classes_csv = args.train_classes

    eval_image_dir = args.eval_dir
    eval_classes_csv = args.eval_classes
    # image_dir = "data/imagenet"
    # classes_csv = "data/imagenet/image_artist.csv"

    # Create dataloader
    train_dataloader, metadata = create_dataloader(
        image_dir=train_image_dir,
        classes_csv=train_classes_csv,
        batch_size=args.batch_size,
    )

    eval_dataloader, metadata = create_dataloader(
        image_dir=eval_image_dir,
        classes_csv=eval_classes_csv,
        batch_size=args.batch_size,
    )

    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(args.input_channels, dropout_rate=0.0).to(device)
    D = Discriminator(args.input_channels).to(device)
    vae = VAEWrapper(args.vae_path).to(device)
    for param in vae.vae.parameters():
        param.detach_()

    # Setup optimizers
    optimizer_G = torch.optim.AdamW(
        G.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(args.beta1, args.beta2)
    )
    optimizer_D = torch.optim.Adam(
        D.parameters(),
        lr=args.lr*2,
        weight_decay=1e-4,
        betas=(args.beta1, args.beta2)
    )
    # scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)
    # scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        start_epoch = checkpoint['epoch']
        G.load_state_dict(checkpoint['generator_state_dict'])
        D.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    else:
        # pretrain generator
        # G = pretrain_generator(G, vae, train_dataloader, device, save_dir, num_epochs=2)
        G.apply(weights_init)
        D.apply(weights_init)
    
    torch.cuda.empty_cache()
    
    for epoch in range(start_epoch, args.num_epochs+1):
        G.train()
        D.train()

        for batch_idx, (real_images, watermark, _) in enumerate(train_dataloader):
            real_images = real_images.to(device)
            watermark = watermark.to(device)

            loss_D_GAN, loss_G_fake, loss_adv, loss_pert = train_step(
                G, D, vae, optimizer_G, optimizer_D,
                real_images, watermark, args
            )

            if batch_idx % 1 == 0:
                print(f"Epoch [{epoch}/{args.num_epochs}] \t"
                      f"Batch [{batch_idx}]\t loss_D_GAN: {loss_D_GAN:.8f} \t"
                      f"loss_G_fake: {loss_G_fake:.8f} \t"
                      f"loss_adv: {loss_adv:.8f} \t"
                      f"loss_pert: {loss_pert:.8f} \t")
        # scheduler_G.step(g_loss['gan_loss'])
        # scheduler_D.step(d_loss) 
        
                
        ### save test image every epoch
        G.eval()
        watermark = create_watermark("vuhoanglong", test_image_size).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        def reverse_transform(tensor):
            # Bước 1: Denormalize ảnh
            tensor = tensor * torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1) + torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            
            # Bước 2: Chuyển từ tensor về dạng PIL Image
            tensor = torch.clamp(tensor, 0, 1)
            
            # Chuyển tensor về dạng PIL Image
            to_pil = transforms.ToPILImage()
            image = to_pil(tensor)
            
            return image
        
        image = transform(test_image).unsqueeze(0).to(device)
        watermark = transform(watermark).unsqueeze(0).to(device)
        perturbation = G(image, watermark)
        adv_image = perturbation + image
        adv_image_clamp = torch.clamp(perturbation, -0.3, 0.3) + image
        adv_image_clamp = torch.clamp(adv_image, 0, 1)
        
        adv_image_ = reverse_transform(adv_image[0].cpu())
        adv_image_.save(f"save_adv_image/adv_image_epoch_{epoch}.png")
        
        adv_image_clamp_ = reverse_transform(adv_image_clamp[0].cpu())
        adv_image_clamp_.save(f"save_adv_image/adv_image_epoch_{epoch}_clamp.png")
        
        # save adv image by 
        # adv_image_resize = reverse_transform(adv_image.squeeze(0).cpu())
        # adv_image_resize.save(f"save_adv_image/adv_image_epoch_{epoch}.png")
        
        diffusion_image = pipe(
            prompt="A painting",
            image=adv_image,
            strength=0.1,
        ).images[0]
        # del pipe
        # diffusion_image = transforms.Resize(test_image_size)(diffusion_image)
        diffusion_image.save(f"save_diffusion_image/diffusion_image_epoch_{epoch}.png")
        
        ### 
        if epoch % 5 == 0 and epoch != 0:
            G.eval()
            adv_metrics = evaluate_adversarial_quality(
                G, eval_dataloader, device)
            print("\nAdversarial Example Quality Metrics:")
            print(f"MSE: {adv_metrics['mse']:.8f}")
            print(f"PSNR: {adv_metrics['psnr']:.8f} dB")
            print(f"SSIM: {adv_metrics['ssim']:.8f}")
        # Save models every 10 epochs
        if (epoch) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': G.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, f"{save_dir}/checkpoint_epoch_{epoch}.pth")


if __name__ == "__main__":
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker = None,
        requires_safety_checker = False,
    ).to('cuda')
    pipe.enable_model_cpu_offload()
    
    args = get_args()
    print(args)
    main(args, pipe)
