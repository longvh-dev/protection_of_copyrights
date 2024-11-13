import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, input_channels=6):  # 3 for image + 3 for watermark
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(input_channels, 64, 2),
            self.conv_block(64, 128, 2),
            self.conv_block(128, 256, 2),
        )
        self.res_blocks = nn.Sequential(*[self.res_block(256) for _ in range(4)])
        self.decoder = nn.Sequential(
            self.deconv_block(256, 128),
            self.deconv_block(128, 64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # Output 3 channels for perturbation
            nn.Tanh()
        )

    def conv_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def res_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, m):
        # Concatenate input image and watermark along channel dimension
        combined_input = torch.cat([x, m], dim=1)
        features = self.encoder(combined_input)
        features = self.res_blocks(features)
        perturbation = self.decoder(features)
        return perturbation