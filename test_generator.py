import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

from utils import create_watermark
from models.generator import Generator
from models.discriminator import Discriminator

class GANInference:
    def __init__(self, checkpoint=None, device='cuda'):
        """
        Initialize inference class with trained models
        
        Args:
            generator_path: Path to generator checkpoint
            discriminator_path: Path to discriminator checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() and device=='cuda' else "cpu")
        
        # Initialize models
        self.generator = Generator(input_channels=6)
        self.discriminator = Discriminator(input_channels=3)
        
        # Load trained weights
        self._load_models(checkpoint)
        
        # Set models to evaluation mode
        self.generator.eval()
        self.discriminator.eval()
        
        # Move models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _load_models(self, checkpoint):
        """Load trained model weights"""
        checkpoint = torch.load(checkpoint, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    def preprocess_image(self, image_path, watermark_text):
        """
        Preprocess input image and watermark
        
        Args:
            image_path: Path to input image
            watermark_text: Text for watermark
        
        Returns:
            Preprocessed image and watermark tensors
        """
        # Load and preprocess input image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Load and preprocess watermark if provided
        watermark = create_watermark(watermark_text, image_size=(256, 256)).convert('RGB')
        watermark.save('watermark.jpg')
        watermark_tensor = self.transform(watermark).unsqueeze(0)
        
        return image_tensor.to(self.device), watermark_tensor.to(self.device)

    def postprocess_image(self, tensor):
        """Convert output tensor to PIL Image"""
        # Denormalize
        tensor = tensor.cpu().squeeze(0)
        tensor = tensor * 0.5 + 0.5
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL Image
        return transforms.ToPILImage()(tensor)

    @torch.no_grad()
    def generate_adversarial(self, image_path, watermark_text, save_path=None):
        """
        Generate adversarial example from input image
        
        Args:
            image_path: Path to input image
            watermark_text: Text to watermark image
            save_path: Path to save output image (optional)
            
        Returns:
            Tuple of (adversarial image, perturbation, discriminator score)
        """
        # Preprocess inputs
        image_tensor, watermark_tensor = self.preprocess_image(image_path, watermark_text)
        
        # Generate perturbation
        perturbation = self.generator(image_tensor, watermark_tensor)
        
        # Create adversarial example
        adversarial = image_tensor + perturbation
        adversarial = torch.clamp(adversarial, -1, 1)
        
        # Get discriminator score
        disc_score = self.discriminator(adversarial)
        
        # Convert outputs to PIL Images
        adv_image = self.postprocess_image(adversarial)
        pert_image = self.postprocess_image(perturbation)
        img = self.postprocess_image(image_tensor)
        watermark = self.postprocess_image(watermark_tensor)
        
        # Save if path provided
        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            filename = os.path.basename(save_path)
            name, ext = os.path.splitext(filename)
            
            adv_image.save(os.path.join(save_dir, f"{name}_adversarial{ext}"))
            pert_image.save(os.path.join(save_dir, f"{name}_perturbation{ext}"))
            img.save(os.path.join(save_dir, f"{name}_image{ext}"))
            watermark.save(os.path.join(save_dir, f"{name}_watermark{ext}"))
        
        return adv_image, pert_image, disc_score

def main():
    # Example usage
    inference = GANInference(
        checkpoint='checkpoints/adv/generator_adv_epoch_50.pth',
        # discriminator_path='checkpoint_epoch_20.pth',
        device='cuda'
    )
    
    # Generate adversarial example
    adv_image, pert_image, disc_score = inference.generate_adversarial(
        image_path='data/test/cat/3156111_a9dba42579.jpg',
        watermark_text='IMAGENET_CAT',
        save_path='output/result.jpg'
    )
    
    # print(f"Discriminator score: {disc_score:.4f}")
    # print(f"Discriminator score: {disc_score}")

if __name__ == '__main__':
    main()