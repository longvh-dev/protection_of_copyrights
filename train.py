import sys
import os
# sys.path.insert(0, os.path.join(os.path.__file__, ".."))

import torch
from torch.utils.data import DataLoader
from datetime import datetime
import os
import argparse
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms


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


def train_step(G, D, vae, optimizer_G, optimizer_D, real_images, watermark, config):
    device = real_images.device
    current_batch_size = real_images.size(0)
    valid = torch.full((current_batch_size, 1), 0.9, device=device)
    fake = torch.full((current_batch_size, 1), 0.1, device=device)


    def _reset_grad():
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
    _reset_grad()
    # Train Discriminator
    fake_images = G(real_images, watermark)
    # fake_images = real_images + perturbation
    d_loss = gan_loss(D(real_images), valid) + gan_loss(D(fake_images.detach()), fake)
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    _reset_grad()
    fake_images = G(real_images, watermark)
    # fake_images = real_images + perturbation

    # objective func
    adv_loss_ = adversarial_loss(vae, fake_images, watermark)
    gan_loss_ = gan_loss(D(fake_images), valid)
    perturbation_loss_ = perturbation_loss(fake_images-real_images, watermark, config.c, config.watermark_region)
    # perturbation_loss_ = 0
    g_loss = adv_loss_ + config.alpha * gan_loss_ + config.beta * perturbation_loss_

    g_loss.backward()
    optimizer_G.step()

    g_loss_ = {'adv_loss': adv_loss_.item(), 'gan_loss': gan_loss_.item(), 'perturbation_loss': perturbation_loss_.item()} 
    return d_loss.item(), g_loss_, fake_images


def main(args, pipe):
    test_image = Image.open('data/wikiart/Early_Renaissance/andrea-del-castagno_dante-alighieri.jpg').convert("RGB")
    test_image_size = test_image.size[::-1]

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
    G = Generator(args.input_channels).to(device)
    D = Discriminator(args.input_channels).to(device)
    vae = VAEWrapper(args.vae_path).to(device)
    for param in vae.vae.parameters():
        param.detach_()

    # Setup optimizers
    optimizer_G = torch.optim.Adam(
        G.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
        betas=(args.beta1, args.beta2)
    )
    optimizer_D = torch.optim.Adam(
        D.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
        betas=(args.beta1, args.beta2)
    )
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch']
        G.load_state_dict(checkpoint['generator_state_dict'])
        D.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    else:
        # pretrain generator
        G.train()
        for epoch in range(5):
            for batch_idx, (real_images, watermark, _) in enumerate(train_dataloader):
                real_images = real_images.to(device)
                watermark = watermark.to(device)
                fake_images = G(real_images, watermark)
                current_batch_size = real_images.size(0)
                g_loss = gan_loss(D(fake_images), torch.full((current_batch_size, 1), 1.0, device=device))
                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()
                if batch_idx % 10 == 0:
                    print(f"Pretrain Epoch [{epoch}/{5}] \t"
                        f"Batch [{batch_idx}] \t"
                        f"g_gan_loss: {g_loss:.4f} \t")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{args.save_dir}/{args.name}/{timestamp}"
    # make save_dir

    os.makedirs(save_dir, exist_ok=True)
    
    
    
    for epoch in range(start_epoch, args.num_epochs+1):
        G.train()
        D.train()

        for batch_idx, (real_images, watermark, _) in enumerate(train_dataloader):
            real_images = real_images.to(device)
            watermark = watermark.to(device)

            d_loss, g_loss, fake_images = train_step(
                G, D, vae, optimizer_G, optimizer_D,
                real_images, watermark, args
            )

            if batch_idx % 1 == 0:
                print(f"Epoch [{epoch}/{args.num_epochs}] \t"
                      f"Batch [{batch_idx}] D_loss: {d_loss:.8f} \t"
                      f"adv_loss: {g_loss['adv_loss']:.8f} \t"
                      f"gan_loss: {g_loss['gan_loss']:.8f} \t"
                      f"perturbation_loss: {g_loss['perturbation_loss']:.8f} \t")
        scheduler_G.step(g_loss['gan_loss'])
        scheduler_D.step(d_loss) 
        
                
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
        # perturbation = G(image, watermark)
        adv_image = G(image, watermark)
        
        adv_image_ = reverse_transform(adv_image[0].cpu())
        adv_image_.save(f"save_adv_image/adv_image_epoch_{epoch}.png")
        
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
