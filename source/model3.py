from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, ResNet18_Weights, ResNet34_Weights
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch import cat
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torchsummary import summary

verb = False


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
    def __init__(self, up_conv_in, out, skip_in):
        super().__init__()

        self.up = nn.ConvTranspose2d(2 * up_conv_in, up_conv_in, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(up_conv_in + skip_in, out)

    def forward(self, inputs, skip):
        if verb: print(f"     input:          {inputs.shape}")
        x = self.up(inputs)
        if verb:
            print(f"     after_up:       {x.shape}")
            print(f"     skip:           {skip.shape}")
        x = cat([x, skip], dim=1)
        if verb: print(f"     up_skip_cat:    {x.shape}")
        x = self.conv(x)
        if verb: print(f"     cat_conv:        {x.shape}\n")
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
        skip_data = dict()
        skip_data[f"layer_0"] = x
        x = self.inputBlock(x)
        skip_data[f"layer_1"] = x
        x = self.inputPool(x)

        for i, block in enumerate(self.downSampleBlocks, 2):
            x = block(x)
            # potřeba nějak změnit, aby to fungovalo, když změníme za jiný resnet
            if i == 5:  # (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            skip_data[f"layer_{i}"] = x

        return x, skip_data


class ourModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_image = BackboneEncoder(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1))
        self.encoder_mask = BackboneEncoder(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1))

        self.bridge = Bridge(1024, 1024)
        """ Decoder """
        self.decoderBlocks = []

        self.decoderBlocks.append(DecoderBlock(512, 512, 512))
        self.decoderBlocks.append(DecoderBlock(256, 256, 256))
        self.decoderBlocks.append(DecoderBlock(128, 128, 128))
        self.decoderBlocks.append(DecoderBlock(64, 64, 128))
        self.decoderBlocks.append(DecoderBlock(32, 32, 6))

        self.outputs = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, x1, x2):
        """
        x1 -- frame with mask
        x2 -- frame
        """
        x1, skip_data_mask = self.encoder_image(x1)
        x2, skip_data_image = self.encoder_mask(x2)

        skip_data_cat = dict()

        if verb:
            print("\n**** ENCODER ****\n")

        for key in skip_data_mask:
            skip_data_cat[key] = cat([skip_data_mask[key], skip_data_image[key]], dim=1)
            if verb:
                print(f"    *** {key} ***")
                print("         skip_mask: " + str(skip_data_mask[key].shape))
                print("         skip_image: " + str(skip_data_image[key].shape))
                print("         concat skip data: " + str(skip_data_cat[key].shape))
                print()

        x = cat([x1, x2], dim=1)

        if verb: print("\n**** BRIDGE ****")
        x = self.bridge(x)

        if verb: print("\n**** DECODER ****")
        for i, block in enumerate(self.decoderBlocks, 1):
            key = f"layer_{5 - i}"
            if verb:
                print(f"*** {key} ***")
                print(block)
            x = block(x, skip_data_cat[key])

        """ Classifier """
        outputs = self.outputs(x)
        return outputs


if __name__ == '__main__':
    BackboneResnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
