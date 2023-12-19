# import cv2
# import segmentation
from dataloader import davis2017Dataset
from torchvision import transforms
import torch.nn as nn
from torch.optim import Adam
from model2 import build_unet
from trainer import Trainer
from torch.utils.data import DataLoader
from lossfunc import DiceLoss, DiceBCELoss, IoULoss
import pickle
from utils import plotLoss
from model3 import ourModel
from model4 import UNetWithResnet50Encoder

if __name__ == '__main__':
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
    # model = build_unet()
    model = ourModel()
    # model = UNetWithResnet50Encoder()

    criterion = nn.BCEWithLogitsLoss()  # DiceLoss()
    lr = 0.001
    epochs = 20

    trainer = Trainer(
        model=model,
        optimizer=Adam(model.parameters(), lr=lr),
        trainingDataloader=trainData,
        validatinDataloader=valData,
        criterion=criterion,
        epochs=epochs
    )

    trainLoss, valLoss = trainer.run()

    model_pkl_file = f"model_lr-{str(lr)}_{epochs}-epochs.pkl"
    with open(model_pkl_file, 'wb') as file:
        pickle.dump(model, file)

    plotLoss(trainLoss, valLoss)
