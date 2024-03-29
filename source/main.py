# import cv2
# import segmentation
from matplotlib import pyplot as plt
import numpy as np
from dataloader import davis2017Dataset, davis2017Datasetv2
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam
from source.eval import evaluation
from trainer import Trainer
from torch.utils.data import DataLoader
from utils import plotLoss
from video_segmentation_model import ourModel
from _old.modelvgg import UNet_vgg

transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_target = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),

])


def evaluate_model(model_file='model_lr-0.002_20-epochs.pth'):
    loaded_model = ourModel()
    loaded_model.load_state_dict(torch.load(model_file))

    trainDataset = davis2017Datasetv2(transform=transform, target_transform=transform_target)

    f1, jaccard = evaluation(model=loaded_model, dataset=trainDataset).evalDavis()
    print(f"f1: {f1}, jaccard: {jaccard}")


def test_model(model_file='model_lr-0.003_20-epochs_0.38_loss.pth'):
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
    plt.imshow(res_numpy, cmap='gray')
    plt.show()
    plt.imshow(img_numpy.transpose(1, 2, 0))
    plt.show()
    plt.imshow(mask_numpy.transpose(1, 2, 0))
    plt.show()


def train_model(lr=0.005, epochs=40):
    """transformCoco = transforms.Compose([
        transforms.Resize(size=(256, 256)),
    ])"""

    # datasets preparation
    trainDataset = davis2017Datasetv2(transform=transform, target_transform=transform_target)
    valDataset = davis2017Dataset(
        dataDir='../datasets/Davis/train480p/DAVIS/',
        annotationsFile='../datasets/Davis/train480p/DAVIS/ImageSets/2017/val.txt',
        transform=transform,
        target_transform=transform_target)
    """
    trainDataset = Coco2017Dataset(transform=transformCoco)
    valDataset = Coco2017Dataset(
        annotationsFile='../datasets/coco2017/raw/instances_val2017.json',
        transform=transformCoco
        )"""

    batch_size = 16

    trainData = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    valData = DataLoader(valDataset, batch_size=batch_size, shuffle=True)

    # models
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

    trainLoss, valLoss, bestModel = trainer.run()

    torch.save(bestModel['models'].state_dict(), f"../models/model_lr-{str(lr)}_{epochs}-epochs_{bestModel['loss']}_loss.pth")
    np.save(f"model_lr-{str(lr)}_{epochs}-valLoss.npz", valLoss)
    np.save(f"model_lr-{str(lr)}_{epochs}-trainLoss.npz", trainLoss)

    plotLoss(trainLoss, valLoss)


def train_model_vgg(lr=0.005, epochs=20):
    transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    transform2 = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),

    ])

    # datasets preparation
    trainDataset = davis2017Dataset(transform=transform, target_transform=transform2)
    valDataset = davis2017Dataset(
        dataDir='../datasets/Davis/train480p/DAVIS/',
        annotationsFile='../datasets/Davis/train480p/DAVIS/ImageSets/2017/val.txt',
        transform=transform,
        target_transform=transform2)

    batch_size = 8

    trainData = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    valData = DataLoader(valDataset, batch_size=batch_size, shuffle=True)

    # models
    model = UNet_vgg()

    criterion = nn.BCEWithLogitsLoss()  # DiceLoss()

    trainer = Trainer(
        model=model,
        optimizer=Adam(model.parameters(), lr=lr),
        trainingDataloader=trainData,
        validatinDataloader=valData,
        criterion=criterion,
        epochs=epochs
    )

    trainLoss, valLoss, bestModel = trainer.run()

    torch.save(bestModel['models'].state_dict(), f"model_lr-{str(lr)}_{epochs}-epochs_{bestModel['loss']}_loss.pth")
    np.save(f"model_lr-{str(lr)}_{epochs}-valLoss.npz", valLoss)
    np.save(f"model_lr-{str(lr)}_{epochs}-trainLoss.npz", trainLoss)

    plotLoss(trainLoss, valLoss)


if __name__ == '__main__':
    # learningRateArray = [0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
    # for l in learningRateArray:
    #    train_model(lr=l)
    #test_model(model_file='model_lr-0.002_20-epochs.pth')

    evaluate_model(model_file='../models/model_lr-0.002_20-epochs.pth')
