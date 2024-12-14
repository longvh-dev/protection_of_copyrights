# import torch
# import torchvision.transforms as transforms
# from PIL import Image

# from models import Generator
# from data_loader import create_watermark

# def reverse_transform(tensor):
#     # Bước 1: Denormalize ảnh
#     # Công thức: x = x * std + mean
#     tensor = tensor * torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1) + torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    
#     # Bước 2: Chuyển từ tensor về dạng PIL Image
#     # Clamp để đảm bảo giá trị nằm trong khoảng [0, 1]
#     tensor = torch.clamp(tensor, 0, 1)
    
#     # Chuyển tensor về dạng PIL Image
#     # Chú ý: Cần chuyển từ (C, H, W) sang (H, W, C)
#     to_pil = transforms.ToPILImage()
#     image = to_pil(tensor)
    
#     return image

# # Ví dụ sử dụng
# # Giả sử generated_image là tensor được sinh ra từ mô hình GAN
# checkpoint = torch.load("checkpoints/dank-base/new/checkpoint_epoch_90.pth")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# G = Generator(3).to(device)
# G.load_state_dict(checkpoint['generator_state_dict'])
# G.eval()
# test_image = Image.open('data/wikiart/Early_Renaissance/andrea-del-castagno_dante-alighieri.jpg').convert("RGB")
# test_image_size = test_image.size[::-1]
# watermark = create_watermark("andrea-del-castagno", test_image_size).convert("RGB")
# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# test_image = transform(test_image).unsqueeze(0).to(device)
# watermark = transform(watermark).unsqueeze(0).to(device)

# adv_image = G(test_image, watermark)
# print(adv_image.shape)

# # Chuyển tensor về dạng ảnh
# adv_image = reverse_transform(adv_image[0].cpu())
# adv_image.save("adv_image.jpg")

import pandas 

df = pandas.read_csv("data/imagenet/train_classes.csv")

# group by artist, for each artist choose 20 images
df = df.groupby('artist').head(20).reset_index(drop=True)
df.to_csv("data/imagenet/train_classes_1.csv", index=False)
