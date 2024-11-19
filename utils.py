import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

def create_watermark(text, font_size=30, image_size=(256, 256), line_spacing=1.25):
    text = text.upper()
    # Tính toán kích thước ảnh mới với padding
    img_width, img_height = image_size
    padded_width = img_width
    padded_height = img_height

    watermark_image = Image.new("RGBA", (padded_width, padded_height), (0, 0, 0, 255))

    # Tạo đối tượng vẽ
    draw = ImageDraw.Draw(watermark_image)

    font = ImageFont.load_default(font_size)  # Sử dụng phông chữ mặc định

    # Tính toán chiều cao của văn bản để lặp lại
    text_bbox = draw.textbbox((0, 0), text, font=font)  # Lấy bounding box của văn bản
    text_height = text_bbox[3] - text_bbox[1]  # Chiều cao văn bản
    line_height = text_height * line_spacing  # Chiều cao dòng với khoảng cách
    num_lines = int(img_height / line_height)  # Số dòng có thể hiển thị


    # Vẽ văn bản lên hình ảnh với padding
    for i in range(num_lines):
        y = i * line_height
        draw.text((2, y), text, fill=(255, 255, 255, 255), font=font)  # Trắng với độ trong suốt

    return watermark_image

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        target_tensor = target_tensor.expand_as(prediction)
        return self.loss(prediction, target_tensor)