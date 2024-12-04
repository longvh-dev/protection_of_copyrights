# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from diffusers import AutoencoderKL

# # Kiến trúc Generator
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU()
#         )
#         self.residual = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, output_padding=1),
#             nn.Tanh()
#         )

#     def forward(self, x, m):
#         combined = torch.cat((x, m), dim=1)
#         encoded = self.encoder(combined)
#         residual = self.residual(encoded)
#         decoded = self.decoder(residual)
#         return decoded

# # Kiến trúc Discriminator
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 1, kernel_size=4, padding=1)
#         )

#     def forward(self, x):
#         return self.model(x)

# # Hàm mất mát
# def adversarial_loss(generator, discriminator, x, m, target_model):
#     x_adv = generator(x, m)
#     pred_real = discriminator(x)
#     pred_fake = discriminator(x_adv)

#     # GAN loss
#     gan_loss = -torch.mean(torch.log(pred_real) + torch.log(1 - pred_fake))

#     # Perturbation loss
#     perturbation = (x_adv - x).abs().mean()
#     perturbation_loss = torch.relu(perturbation - 10/255)

#     # Adversarial loss
#     target_output_adv = target_model(x_adv)
#     target_output_m = target_model(m)
#     adv_loss = torch.norm(target_output_adv - target_output_m, p=2)

#     return gan_loss + 10 * perturbation_loss + adv_loss

# # Huấn luyện
# def train(generator, discriminator, dataloader, target_model, epochs=10, lr=1e-4):
#     gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
#     disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

#     for epoch in range(epochs+1):
#         for x, m in dataloader:  # x: input image, m: watermark
#             x_adv = generator(x, m)

#             # Cập nhật discriminator
#             disc_optimizer.zero_grad()
#             real_loss = -torch.mean(discriminator(x))
#             fake_loss = torch.mean(discriminator(x_adv.detach()))
#             disc_loss = real_loss + fake_loss
#             disc_loss.backward()
#             disc_optimizer.step()

#             # Cập nhật generator
#             gen_optimizer.zero_grad()
#             loss = adversarial_loss(generator, discriminator, x, m, target_model)
#             loss.backward()
#             gen_optimizer.step()

#         print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
#         if epoch % 5 == 0:
#             torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")
#             torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch}.pth")

# # Tạo đối tượng model
# generator = Generator()
# discriminator = Discriminator()
# target_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

from PIL import Image, ImageDraw, ImageFont
from data_loader import create_watermark
from torchvision import transforms
from diffusers import StableDiffusionImg2ImgPipeline
import torch

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker = None,
    requires_safety_checker = False,
).to('cuda')
pipe.enable_model_cpu_offload()

test_image = Image.open('data/wikiart/Early_Renaissance/andrea-del-castagno_dante-alighieri.jpg').convert("RGB")
test_image_size = test_image.size[::-1]
# watermark = create_watermark("vuhoanglong", test_image.size).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
reverse_transform = transforms.Compose([
    transforms.Normalize(mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0)),
    transforms.ToPILImage(),
    transforms.Resize(test_image_size),
])
test_image = transform(test_image).unsqueeze(0).to("cuda")
# watermark = transform(watermark).unsqueeze(0).to("cuda")

        # save adv image by 
adv_image_resize = reverse_transform(test_image.squeeze(0).cpu())
adv_image_resize.save(f"save_adv_image/adv_image_epoch_.png")

diffusion_image = pipe(
    prompt="A painting",
    image=test_image,
    strength=0.1,
).images[0]
del pipe
diffusion_image = transforms.Resize(test_image_size)(diffusion_image)
diffusion_image.save(f"save_diffusion_image/diffusion_image_epoch_.png")