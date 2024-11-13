import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random

from utils import create_watermark

class WatermarkDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256)):
        self.data_dir = data_dir
        self.image_files = []
        self.image_size = image_size

        # Collect all image files from the directory structure
        for category in os.listdir(data_dir):
            if category.startswith('.'):
                continue
            
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path):
                for image_file in os.listdir(category_path):
                    try:
                        if image_file.startswith('.'):
                            continue
                        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                          Image.open(os.path.join(category_path, image_file)).verify()
                    except (IOError, SyntaxError):
                        print(f"Invalid image: {image_file}")
                        continue
                    self.image_files.append(os.path.join(category, image_file))


        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])



    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and transform image
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.image_transform(image)

        # Create watermark
        text = img_path.split(os.path.sep)[-2]
        watermark = create_watermark(text, image_size=self.image_size).convert('RGB')
        watermark = self.image_transform(watermark)

        return image, watermark

def get_dataloader(data_dir, batch_size=8, num_workers=2):
    dataset = WatermarkDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

