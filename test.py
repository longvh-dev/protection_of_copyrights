# import torch
# import requests
# from PIL import Image
# from io import BytesIO

# from diffusers import StableDiffusionImg2ImgPipeline

# # load the pipeline
# device = "cuda"
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16
# ).to(device)

# # let's download an initial image
# url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

# response = requests.get(url)
# init_image = Image.open(BytesIO(response.content)).convert("RGB")
# init_image.thumbnail((768, 768))

# prompt = "A fantasy landscape, trending on artstation"

# images = pipe(prompt=prompt, init_image=init_image, strength=0.01, guidance_scale=7.5).images

# images[0].save("fantasy_landscape.png")

# from utils import create_watermark

# watermark = create_watermark("albrecht durer", image_size=(256, 256)).convert('RGB')
# watermark.save('watermark.jpg')

import torch
from PIL import Image
from models.discriminator import Discriminator
from torchvision import transforms
from utils import GANLoss

checkpoint = 'checkpoints/train_10_samples/checkpoint_epoch_20.pth'
device = 'cuda'

adv_image = Image.open('output/result_image.jpg').convert('RGB')

discriminator = Discriminator(input_channels=3).to(device)
gan_loss = GANLoss().to(device)
checkpoint = torch.load(checkpoint, map_location=device)
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

discriminator.eval()

# Output should be close to 0 for fake images
with torch.no_grad():
    adv_image_tensor = transforms.ToTensor()(adv_image).unsqueeze(0).to(device)
    output = discriminator(adv_image_tensor)

    # Apply sigmoid activation to interpret the output as a probability
    d_loss = gan_loss(output, False)

    print(d_loss)