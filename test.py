from PIL import Image
from data_loader import create_watermark

img_path = '/home/pdlong/copyrights/data/wikiart/Abstract_Expressionism/aaron-siskind_chicago-6-1961.jpg'
image = Image.open(img_path).convert('RGB')
image_size = image.size  # Save original image size

# print(image_size)
watermark = create_watermark(
    "aaron-siskind", image_size=image_size).convert('RGB')

watermark.save('watermark.png')
