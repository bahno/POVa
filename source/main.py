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
"""
# Replace 'video_file_path' with the path to your video file
video_file_path = '../data/travolta.gif'

# Open the video file
cap = cv2.VideoCapture(video_file_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

prev_frame = None

# Read and display frames until the video is over
while True:
    # Read a frame from the video
    ret, frame = cap.read()
"""
"""
    if not prev_frame:
        #TODO manually create initial mask for the first processed frame
        pass
    """
"""
    # Check if the frame is read successfully
    if not ret:
        print("End of video")
        break

    display_frame = segmentation.perform_segmentation(frame, prev_frame)

    # Display the frame
    cv2.imshow('Original gif', display_frame)

    prev_frame = display_frame

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
"""

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
    ])

    # datasets preparation
    trainDataset = davis2017Dataset(transform=transform)
    valDataset = davis2017Dataset(
        #gtDir='../datasets/Davis/train480p/DAVIS/Annotations/480p/',
        dataDir='../datasets/Davis/train480p/DAVIS/JPEGImages/480p/',
        annotationsFile='../datasets/Davis/train480p/DAVIS/ImageSets/2017/val.txt',
        transform=transform)

    batch_size = 4

    trainData = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    valData = DataLoader(valDataset, batch_size=batch_size, shuffle=True)

    # model
    #model = build_unet()
    model = UNetWithResnet50Encoder()

    criterion = nn.BCEWithLogitsLoss()#DiceLoss()
    lr = 0.001

    trainer = Trainer(
        model=model,
        optimizer=Adam(model.parameters(), lr=lr),
        trainingDataloader=trainData,
        validatinDataloader=valData,
        criterion=criterion,
        epochs=2
    )
    
    trainLoss, valLoss = trainer.run()

    model_pkl_file = "model_" + str(lr) + ".pkl"
    with open(model_pkl_file, 'wb') as file:
        pickle.dump(model, file)

    plotLoss(trainLoss, valLoss)
