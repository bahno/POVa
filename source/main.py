# import cv2
# import segmentation
from matplotlib import pyplot as plt

from dataloader import davis2017Dataset
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam
from model2 import build_unet
from trainer import Trainer
from torch.utils.data import DataLoader
from lossfunc import DiceLoss, DiceBCELoss, IoULoss
from torch import load
import pickle
from utils import plotLoss
from model3 import ourModel
from model4 import UNetWithResnet50Encoder


def test_model(model_file='model_lr-0.002_20-epochs.pth'):
    loaded_model = ourModel()
    loaded_model.load_state_dict(torch.load(model_file))
    loaded_model.eval()

    # Define the transformation for the input image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])
    # Load the images from a file path
    image_path = '../datasets/Davis/train480p/DAVIS/JPEGImages/480p/bear/00001.jpg'
    mask_path = '../datasets/Davis/train480p/DAVIS/Merged/480p/bear/00000.jpg'

    # Apply the defined transformation
    img_tensor = transform(Image.open(image_path))
    mask_tensor = transform(Image.open(mask_path))

    # Add a batch dimension (if needed)
    img_tensor_uns = img_tensor.unsqueeze(0)
    mask_tensor_uns = mask_tensor.unsqueeze(0)

    res = loaded_model.forward(img_tensor_uns, mask_tensor_uns)

    res_thresh = torch.threshold(res, -1.5, 1)

    # Convert the PyTorch tensor to a NumPy array
    res_thresh = res_thresh.detach().squeeze().numpy()
    res_numpy = res.detach().squeeze().numpy()
    img_numpy = img_tensor.detach().squeeze().numpy()
    mask_numpy = mask_tensor.detach().squeeze().numpy()

    # Display the NumPy array using Matplotlib
    plt.imshow(res_thresh, cmap='gray')
    plt.show()
    plt.imshow(res_numpy, cmap='gray')
    plt.show()
    plt.imshow(img_numpy.transpose(1, 2, 0))
    plt.show()
    plt.imshow(mask_numpy.transpose(1, 2, 0))
    plt.show()


def train_model(lr=0.004, epochs=20):

    transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
    ])

    # datasets preparation
    trainDataset = davis2017Dataset(transform=transform)
    valDataset = davis2017Dataset(
        # gtDir='../datasets/Davis/train480p/DAVIS/Annotations/480p/',
        dataDir='../datasets/Davis/train480p/DAVIS/JPEGImages/480p/',
        annotationsFile='../datasets/Davis/train480p/DAVIS/ImageSets/2017/val.txt',
        transform=transform)

    batch_size = 4

    trainData = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    valData = DataLoader(valDataset, batch_size=batch_size, shuffle=True)

    # model
    model = ourModel()

    criterion = nn.BCEWithLogitsLoss()  # DiceLoss()

    trainer = Trainer(
        model=model,
        optimizer=Adam(model.parameters(), lr=lr),
        trainingDataloader=trainData,
        validatinDataloader=valData,
        criterion=criterion,
        epochs=epochs
    )

    trainLoss, valLoss = trainer.run()

    torch.save(model.state_dict(), f"model_lr-{str(lr)}_{epochs}-epochs.pth")

    plotLoss(trainLoss, valLoss)


if __name__ == '__main__':
    train_model()
    # test_model()
