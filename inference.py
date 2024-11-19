import torch
from torchvision import transforms
from diffusers import StableDiffusionImg2ImgPipeline
from models.generator import Generator
from PIL import Image
from data_loader import create_watermark
import warnings

warnings.filterwarnings("ignore")

def test_with_diffusion(generator, image, watermark, prompt, strength=0.1):
    # Generate adversarial image
    with torch.no_grad():
        perturbation = generator(image, watermark)
        adversarial_image = image + perturbation
    
    # Test with Stable Diffusion
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to('cuda')
    # pipe.enable_model_cpu_offload()
    print("loaded model")
    
    # Generate from adversarial image
    output = pipe(
        prompt=prompt,
        image=adversarial_image,
        strength=strength,
    ).images[0]
    # del pipe
    
    return output, image

def test_without_generator(image, prompt, strength=0.1):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        force_download=True
    ).to('cuda')
    pipe.enable_model_cpu_offload()
    print("loaded model")

    # Generate from original image
    output = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
    ).images[0]
    # del pipe
    return output, image

def main():
    generator = Generator().to('cuda')
    checkpoint = torch.load('checkpoints/20241112-221929/checkpoint_epoch_200.pth')
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    # Test cases
    test_prompts = [
        "A photo of base image",
        # "A oil painting of base image",
    ]
    print(test_prompts)

    image = Image.open("data/wikiart/Early_Renaissance/andrea-del-castagno_crucifixion-1.jpg").convert("RGB")
    image_size = image.size  # Save original image size
    print(image_size)
    watermark = create_watermark("aaron-siskind", image_size).convert("RGB")

    # convert to tensor
    transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    resize = transforms.Resize((1382, 1732))
    image_ = transform(image).unsqueeze(0).to('cuda')
    watermark_ = transform(watermark).unsqueeze(0).to('cuda')

    for prompt in test_prompts:
        output, _ = test_with_diffusion(generator, image_, watermark_, prompt, strength=0.1)
        output_resized = resize(output)
        # save output resized
        output_resized.save(f"output/resize_{prompt}.png")
        
        output.save(f"output/{prompt}.png")

    for prompt in test_prompts:
        output, _ = test_without_generator(image, prompt, strength=0.1)
        # Save or analyze results
        output.save(f"output/{prompt}_without_generator.png")

if __name__ == "__main__":
    main()
