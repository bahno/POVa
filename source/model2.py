from torch import nn
from torchvision.transforms import CenterCrop, Compose, ToTensor, Resize
from torch.nn import functional as F
import torch
import torch.optim as optim
from dataloader import davis2017Dataset
from torch.utils.data import DataLoader


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


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))     
    
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
    
class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)         
        """ Bottleneck """
        self.b = conv_block(512, 1024)         
        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)         
        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)    

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)         
        """ Bottleneck """
        b = self.b(p4)         
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)         
        """ Classifier """
        outputs = self.outputs(d4)        
        return outputs

"""
model = build_unet()  # předpokládáme RGB obrázky na vstupu a binární segmentaci na výstupu
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Příklad trénování s daty
# DataLoader a další úpravy by byly potřeba podle konkrétního datasetu
transform = Compose([
        Resize(size=(256,256)),
        ToTensor(),
    ])
trainDataset = davis2017Dataset(transform=transform)
testDataset  = davis2017Dataset(
    				gtDir = '../datasets/Davis/test480p/DAVIS',
                	dataDir = '../datasets/Davis/test480p/DAVIS',
                    annotationsFile = '../datasets/Davis/test480p/DAVIS'
                    transform=transform)
batch_size = 8


trainData = DataLoader(dataset, batch_size=batch_size, shuffle=True)
testData  = DataLoader()


num_epochs = 100
# Iterace přes epochy
for epoch in range(num_epochs):
    # Iterace přes dávky (batche) dat
    for inputs, labels in trainData:
        optimizer.zero_grad()  # Nastavení gradientů na nulu
        outputs = model(inputs)  # Předpovědi modelu
        loss = criterion(outputs, labels)  # Výpočet ztrát
        loss.backward()  # Zpětný průchod
        optimizer.step()  # Aktualizace vah


    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()},') # Validation Loss: {validation_loss}


"""


