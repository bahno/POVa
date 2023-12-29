import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import  VGG19_BN_Weights

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(2 * in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()
        vgg = models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        self.features_image = vgg.features
        self.features_mask = vgg.features


    def forward(self, x_image, x_mask):
        x_image = self.features_image(x_image)
        x_mask = self.features_mask(x_mask)

        return x_image, x_mask



class UNet_vgg(nn.Module):
    def __init__(self):
        super(UNet_vgg, self).__init__()
        self.encoder = EncoderBlock()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            DecoderBlock(512, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            ConvBlock(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            ConvBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ConvBlock(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            ConvBlock(32, 32),
            nn.Conv2d(32, 1, kernel_size=1)  # Adjust the output channels to match the target size
            
        )

    def forward(self, x_image, x_mask):
        x_image, x_mask = self.encoder(x_image, x_mask)
        x = torch.cat((x_image, x_mask), dim=1)
        x = self.decoder(x)
        return x