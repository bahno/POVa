import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


def plotLoss(trainLoss : [], 
            valLoss : []):
    
    epochs = np.array(range(1,len(trainLoss)+1))
    
    plt.plot(epochs, trainLoss, label='Training Loss')
    plt.plot(epochs, valLoss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, len(trainLoss)+5, 2))

    # Display the plot
    plt.legend(loc='best')
    plt.show()


def testCocoImages(folder_path = '../datasets/coco2017/train/'): 
        extensions = []
        for fldr in os.listdir(folder_path):
                sub_folder_path = os.path.join(folder_path, fldr)
                print(sub_folder_path)
                for filee in os.listdir(sub_folder_path):
                        file_path = os.path.join(sub_folder_path, filee)
                        print('** Path: {}  **'.format(file_path), end="\r", flush=True)
                        im = Image.open(file_path)
                        rgb_im = im.convert('RGB')
                        if filee.split('.')[1] not in extensions:
                                extensions.append(filee.split('.')[1])


def accuracy(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask==pred_mask)
    acc = np.mean(xor/(union + xor - intersect))
    return round(acc, 3)

if __name__ == '__main__':
       testCocoImages()