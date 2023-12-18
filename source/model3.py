from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, ResNet18_Weights, ResNet34_Weights
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch import cat
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torchsummary import summary


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
    def __init__(self, up_conv_in, out, skip_in=0):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(up_conv_in, out, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(up_conv_in, out)

    def forward(self, inputs, skip):
        print(f"     input:          {inputs.shape}")
        x = self.up(inputs)
        print(f"     after_up:       {x.shape}")
        print(f"     skip:           {skip.shape}")
        x = cat([x, skip], dim=1)
        print(f"     up_skip_cat:    {x.shape}")

        x = self.conv(x)
        print(f"     cat_conv:        {x.shape}\n")
        return x


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class BackboneEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder1 = encoder

        downSampleBlocks = []
        self.inputBlock = nn.Sequential(*list(encoder.children()))[:3]
        self.inputPool = list(encoder.children())[3]

        for bottleneck in list(self.encoder1.children()):
            if isinstance(bottleneck, nn.Sequential):
                downSampleBlocks.append(bottleneck)
        self.downSampleBlocks = nn.ModuleList(downSampleBlocks)

    def forward(self, x):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.inputBlock(x)
        pre_pools[f"layer_1"] = x
        x = self.inputPool(x)

        for i, block in enumerate(self.downSampleBlocks, 2):
            x = block(x)
            # potřeba nějak změnit, aby to fungovalo, když změníme za jiný resnet
            if i == 5:  # (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        return x, pre_pools


class ourModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_image = BackboneEncoder(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1))
        self.encoder_mask = BackboneEncoder(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1))

        self.bridge = Bridge(1024, 1024)
        """ Decoder """
        self.decoderBlocks = []

        # L4
        # in: 1024 from bridge, 512 from skip L4
        # out: 512
        self.decoderBlocks.append(DecoderBlock(1024, 512, 512))

        # L3
        # in: 512 from L4, 256 from skip L3
        # out: 256
        self.decoderBlocks.append(DecoderBlock(512, 256, 256))

        # L2
        # in: 256 from L3, 128 from skip L2
        # out: 128
        self.decoderBlocks.append(DecoderBlock(256, 128, 128))

        # L1
        # in: 128 from L2, 64 from skip L1
        # out: 64
        self.decoderBlocks.append(DecoderBlock(128, 64, 64))

        # L0
        # in: 64 from bridge, 32 from skip l1
        # out: 32
        self.decoderBlocks.append(DecoderBlock(64, 32, 32))

        self.outputs = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, x1, x2):
        """
        x1 -- frame with mask
        x2 -- frame
        """
        x1, skip_data_mask = self.encoder_image(x1)
        x2, skip_data_image = self.encoder_mask(x2)

        skip_data_cat = dict()

        print("\n**** ENCODER ****")
        for key in skip_data_mask:
            print("     skip_mask: " + str(skip_data_mask[key].shape))
            print("     skip_image: " + str(skip_data_image[key].shape))
            skip_data_cat[key] = cat([skip_data_mask[key], skip_data_image[key]], dim=1)
            print("     concat skip data: " + str(skip_data_cat[key].shape))
            print()

        x = cat([x1, x2], dim=1)

        print("\n**** BRIDGE ****")
        x = self.bridge(x)

        print("\n**** DECODER ****")
        for i, block in enumerate(self.decoderBlocks, 1):
            key = f"layer_{5 - i}"
            print(f"*** {key} ***")
            x = block(x, skip_data_cat[key])

        """ Classifier """
        outputs = self.outputs(x)
        return outputs


if __name__ == '__main__':
    BackboneResnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
