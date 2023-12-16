from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, ResNet18_Weights, ResNet34_Weights
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch import cat
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torchsummary import summary


class conv_block(nn.Module):
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



class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    
class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            conv_block(in_channels, out_channels),
            conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)

class BackboneEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder1 = encoder
        
        self.downSampleBlocks = []
        self.inputBlock = nn.Sequential(*list(encoder.children()))[:3]
        self.inputPool = list(encoder.children())[3]
        for bottleneck in list(self.encoder1.children()):
            if isinstance(bottleneck, nn.Sequential):
                self.downSampleBlocks.append(bottleneck)

    
    def forward(self, x):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.inputBlock(x)
        

        pre_pools[f"layer_1"] = x
        x = self.inputPool(x)
        for i, block in enumerate(self.downSampleBlocks, 2):
            x = block(x)
            #potřeba nějak změnit, aby to fungovalo, když změníme za jiný resnet
            if i == 5: #(UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x


        return x, pre_pools




class ourModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = BackboneEncoder(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1))
        self.e2 = BackboneEncoder(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1))
        
    


        self.b = Bridge(1024, 1024)               
        """ Decoder """
        self.decoderBlocks = []
        self.decoderBlocks.append(decoder_block(1024, 512))
        self.decoderBlocks.append(decoder_block(512, 256))
        self.decoderBlocks.append(decoder_block(256, 128))
        self.decoderBlocks.append(decoder_block(128, 64))
        self.decoderBlocks.append(decoder_block(64, 32))

        self.outputs = nn.Conv2d(32, 1, kernel_size=1, padding=0)    


    def forward(self, x1, x2):
        """
        x1 -- frame with mask
        x2 -- frame
        """
        x1, prePoolsX1 = self.e1(x1)
        x2, prePoolsX2 = self.e2(x2)
        for key in prePoolsX1:
            prePoolsX1[key] = cat([prePoolsX1[key],prePoolsX2[key]],dim=1)
            

        x = cat([x1,x2],dim=1)
        x = self.b(x)         

        for i, block in enumerate(self.decoderBlocks, 1): 
            key = f"layer_{5 - i}"
           
            x = block(x, prePoolsX1[key])





        """ Classifier """
        outputs = self.outputs(x)        
        return outputs




if __name__ == '__main__':
    BackboneResnet18 = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1) 
   