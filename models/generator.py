import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.instance_norm1 = nn.InstanceNorm2d(channels)
        self.instance_norm2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.relu(self.instance_norm1(self.conv1(x)))
        x = self.instance_norm2(self.conv2(x))
        return x + residual

class Generator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.encoders = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            # state size. 64 x 256 x 256
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            # state size. 128 x 128 x 128
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            # state size. 256 x 64 x 64
        )
        
        self.residual_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(4)])
        
        self.decoders = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            # state size. 128 x 64 x 64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            # state size. 64 x 128 x 128
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            # state size. 32 x 256 x 256
        )

        self.final = nn.Sequential(
            nn.Conv2d(32, input_channels, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x, m):
        combined = torch.cat([x, m], dim=1)
        out = self.encoders(combined)
        out = self.residual_blocks(out)
        out = self.decoders(out)
        out = self.final(out)
        
        # Ensure output matches input spatial dimensions
        assert out.shape[2:] == x.shape[2:], f"Output shape {out.shape} doesn't match input shape {x.shape}"
        
        return out