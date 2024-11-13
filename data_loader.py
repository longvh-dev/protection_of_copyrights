# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import os
# import random

# from utils import create_watermark

# class WatermarkDataset(Dataset):
#     def __init__(self, data_dir, image_size=(256, 256)):
#         self.data_dir = data_dir
#         self.image_files = []
#         self.image_size = image_size

#         # Collect all image files from the directory structure
#         for category in os.listdir(data_dir):
#             if category.startswith('.'):
#                 continue
            
#             category_path = os.path.join(data_dir, category)
#             if os.path.isdir(category_path):
#                 for image_file in os.listdir(category_path):
#                     try:
#                         if image_file.startswith('.'):
#                             continue
#                         if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                           Image.open(os.path.join(category_path, image_file)).verify()
#                     except (IOError, SyntaxError):
#                         print(f"Invalid image: {image_file}")
#                         continue
#                     self.image_files.append(os.path.join(category, image_file))


#         self.image_transform = transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])



#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         # Load and transform image
#         img_path = os.path.join(self.data_dir, self.image_files[idx])
#         image = Image.open(img_path).convert('RGB')
#         image = self.image_transform(image)


#         # Create watermark
#         text = img_path.split(os.path.sep)[-2]
#         watermark = create_watermark(text, image_size=self.image_size).convert('RGB')
#         watermark = self.image_transform(watermark)

#         # add noise to watermark and image to avoid overfitting
#         # Set seed for reproducibility
#         seed = 42
#         torch.manual_seed(seed)
#         random.seed(seed)

#         # Add noise to watermark and image to avoid overfitting
#         watermark = add_salt_and_pepper_noise(watermark, prob=0.05)
#         watermark = add_gaussian_blur(watermark, kernel_size=3, sigma=0.8)
        
#         # image = add_poisson_noise(image)
#         image = add_speckle_noise(image, noise_level=0.05)

#         return image, watermark

# def get_dataloader(data_dir, batch_size=8, num_workers=2):
#     dataset = WatermarkDataset(data_dir)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# def add_salt_and_pepper_noise(img, prob=0.01):
#     """Add salt and pepper noise to an image tensor."""
#     noisy = img.clone()
#     mask = torch.rand_like(img) < prob
#     noisy[mask] = torch.where(torch.rand_like(noisy[mask]) < 0.5, 0.0, 1.0)
#     return noisy

# def add_poisson_noise(img):
#     """Add Poisson noise to an image tensor."""
#     noisy = img + torch.randn_like(img) * torch.sqrt(img)
#     return torch.clamp(noisy, 0, 1)

# def add_gaussian_blur(img, kernel_size=5, sigma=1.0):
#     """Add Gaussian blur to an image tensor."""
#     transform = transforms.GaussianBlur(kernel_size, sigma)
#     return transform(img)

# def add_speckle_noise(img, noise_level=0.1):
#     """Add speckle noise to an image tensor."""
#     noise = torch.randn_like(img) * noise_level
#     return torch.clamp(img + img * noise, 0, 1)


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from typing import List, Tuple, Optional, Dict
from PIL import ImageDraw, ImageFont

class WatermarkDataset(Dataset):
    """Custom Dataset for watermark training that uses artist names as watermarks"""
    
    def __init__(
        self,
        image_dir: str,
        classes_csv: str,
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (256, 256)
    ):
        """
        Args:
            image_dir (str): Directory with all the images
            classes_csv (str): Path to CSV file containing filename, artist, genre
            transform (transforms.Compose, optional): Optional transform to be applied on images
            image_size (tuple): Size to resize images to
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform
        
        # Load and process the classes CSV
        self.metadata_df = pd.read_csv(classes_csv)
        
        # Verify that the images exist
        self.valid_files = []
        for idx, row in self.metadata_df.iterrows():
            filepath = os.path.join(image_dir, row['filename'])
            if os.path.exists(filepath):
                self.valid_files.append(idx)
            
        # Create mappings for artists and genres
        self.artists = sorted(self.metadata_df['artist'].unique())
        self.genres = sorted(self.metadata_df['genre'].unique())
        self.artist_to_idx = {artist: idx for idx, artist in enumerate(self.artists)}
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
        # self.transform = transform

    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Get metadata for this index
        metadata_idx = self.valid_files[idx]
        row = self.metadata_df.iloc[metadata_idx]
        
        # Load and transform image
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Use artist name as watermark text
        artist_name = row['artist']
        watermark = create_watermark(artist_name, self.image_size).convert('RGB')
        
        # Convert watermark to tensor if it isn't already
        if not isinstance(watermark, torch.Tensor):
            watermark = transforms.ToTensor()(watermark)
        
        # Create metadata dictionary
        metadata = {
            'filename': row['filename'],
            'artist': artist_name,
            'genre': row['genre'],
            'artist_idx': self.artist_to_idx[artist_name],
            'genre_idx': self.genre_to_idx[row['genre']],
            # 'watermark': watermark.shape,
        }
            
        return image, watermark, metadata

def create_dataloader(
    image_dir: str,
    classes_csv: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    shuffle: bool = True
) -> Tuple[DataLoader, Dict]:
    """
    Creates a DataLoader for the watermark training pipeline
    
    Args:
        image_dir (str): Directory containing training images
        classes_csv (str): Path to CSV file containing filename, artist, genre
        batch_size (int): Number of samples per batch
        num_workers (int): Number of subprocesses for data loading
        image_size (tuple): Size to resize images to
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        Tuple[DataLoader, Dict]: PyTorch DataLoader object and metadata dictionary
    """
    dataset = WatermarkDataset(
        image_dir=image_dir,
        classes_csv=classes_csv,
        image_size=image_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create metadata dictionary with mappings
    metadata = {
        'num_artists': len(dataset.artists),
        'num_genres': len(dataset.genres),
        'artist_to_idx': dataset.artist_to_idx,
        'genre_to_idx': dataset.genre_to_idx,
        'idx_to_artist': {v: k for k, v in dataset.artist_to_idx.items()},
        'idx_to_genre': {v: k for k, v in dataset.genre_to_idx.items()},
        'artists': dataset.artists,
        'genres': dataset.genres
    }
    
    return dataloader, metadata

def create_watermark(
    text: str,
    image_size: Tuple[int, int] = (256, 256),
    font_size: int = 30,
    line_spacing: float = 1.25
) -> Image.Image:
    """
    Creates a watermark image with repeated text
    
    Args:
        text (str): Text to use as watermark
        image_size (tuple): Size of output image (width, height)
        font_size (int): Font size for watermark text
        line_spacing (float): Spacing between text lines
        
    Returns:
        Image.Image: PIL Image containing the watermark
    """
    
    text = text.upper()
    img_width, img_height = image_size
    
    # Create new transparent image
    watermark_image = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(watermark_image)
    
    # Set up font
    # Adjust font size based on text length
    adjusted_font_size = max(10, int(font_size * (15 / (len(text) + 5))))  # Minimum size of 10
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", adjusted_font_size)
    except:
        font = ImageFont.load_default()
    
    # Calculate text dimensions and spacing
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_height = text_bbox[3] - text_bbox[1]
    line_height = text_height * line_spacing
    num_lines = int(img_height / line_height)
    
    # Draw watermark text lines
    for i in range(num_lines):
        y = i * line_height
        draw.text((2, y), text, fill=(255, 255, 255, 255), font=font)
        
    return watermark_image

# Example usage:
if __name__ == "__main__":
    # Example parameters
    image_dir = "data/wikiart"
    classes_csv = "data/wikiart/train_classes.csv"
    
    # Create dataloader
    dataloader, metadata = create_dataloader(
        image_dir=image_dir,
        classes_csv=classes_csv,
        batch_size=32
    )
    
    # Print dataset information
    print(f"Number of artists: {metadata['num_artists']}")
    print(f"Number of genres: {metadata['num_genres']}")
    print(f"Available artists: {metadata['artists'][:5]}...")  # Show first 5 artists
    # print(f'Watermark shape: {metadata["watermark"]}')
    
    # Test the dataloader
    # for images, watermarks, batch_metadata in dataloader:
    #     print(f"Batch shapes: Images {images.shape}, Watermarks {watermarks.shape}")
    #     print(f"Sample metadata: {batch_metadata['filename'][0]}, "
    #           f"Artist: {batch_metadata['artist'][0]}, "
    #           f"Genre: {batch_metadata['genre'][0]}")
    #         #   f"Watemark shape: {batch_metadata['watermark']}")
    #     break  # Just test one batch

    for batch_idx, (images, watermarks, batch_metadata) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        # print(f"Images shape: {images.shape}")
        # print(f"Watermarks shape: {watermarks.shape}")
        # print(f"Metadata: {batch_metadata}")
        # break
