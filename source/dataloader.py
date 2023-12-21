import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from pycocotools.coco import COCO
from PIL import Image
import json
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
from pycocotools.coco import COCO
from matplotlib import image
from pathlib import Path
from matplotlib import pyplot as plt
from data_augmentation import blend_image_mask
from torchvision.transforms import v2
import random
import torchvision.transforms.functional as TF
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class davis2017Dataset(Dataset):
    def __init__(self,
                 # gtDir='../datasets/Davis/train480p/DAVIS/Annotations/480p/',
                 dataDir='../datasets/Davis/train480p/DAVIS/JPEGImages/480p/',
                 annotationsFile='../datasets/Davis/train480p/DAVIS/ImageSets/2017/train.txt',

                 seqNum=0,
                 transform=None,
                 target_transform=None):
        self.dataDir = dataDir
        self.gtDir = dataDir.replace("JPEGImages", "Annotations")
        self.mergedDir = dataDir.replace("JPEGImages", "Merged")
        self.imgDirs = pd.read_csv(annotationsFile, header=None, names=["ImageDirNames"])
        self.transform = transform
        self.target_transform = target_transform
        self.seqNum = str(seqNum).zfill(5)
        self.nextSeqNum = str(seqNum + 1).zfill(5)

    def __len__(self):
        return len(self.imgDirs)

    def __getitem__(self, idx):
        currImg = Image.open(
            os.path.join(self.mergedDir, self.imgDirs.at[idx, "ImageDirNames"], self.seqNum + '.jpg').replace(os.sep,
                                                                                                              '/')).convert(
            "RGB")
        prevImage = Image.open(
            os.path.join(self.dataDir, self.imgDirs.at[idx, "ImageDirNames"], self.nextSeqNum + '.jpg').replace(os.sep,
                                                                                                                '/')).convert(
            "RGB")
        gt = np.array(Image.open(
            os.path.join(self.gtDir, self.imgDirs.at[idx, "ImageDirNames"], self.nextSeqNum + '.png').replace(os.sep,
                                                                                                              '/')).convert(
            "L"),
                      dtype=np.float32)

        gt = ((gt / np.max([gt.max(), 1e-8])) > 0.5).astype(np.float32)

        gt = Image.fromarray(np.uint8((gt) * 255))
        if self.transform is not None:
            currImg = self.transform(currImg)
            prevImage = self.transform(prevImage)
            gt = self.target_transform(gt)


        return prevImage, currImg, gt




class Coco2017Dataset(Dataset):
    def __init__(self, 
                datasetDir = '../datasets/coco2017/train/data/', 
                annotationsFile = '../datasets/coco2017/raw/instances_train2017.json',
                transform = None):
        self.datasetDir = datasetDir
        self.annFileTrain = annotationsFile
        self.coco = COCO(annotationsFile)
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.imgIds = self.coco.getImgIds(catIds=self.cat_ids)
        self.transform = transform
     

    def __len__(self):
        return len(self.imgIds)


    def __getitem__(self, idx):
        imgId = self.imgIds[idx]
        imgObj = self.coco.loadImgs(imgId)[0]
        annsObj = self.coco.loadAnns(self.coco.getAnnIds(imgId)) 

        img = Image.open(os.path.join(self.datasetDir, imgObj['file_name']))


        mask = self.coco.annToMask(ann=annsObj[0])
        for i in range(len(annsObj)):
            mask |= self.coco.annToMask(annsObj[i])
        mask = mask * 255

        im = Image.fromarray(mask)
        currentImage = None

        if (imgId == 86): 
            img.show()
        prevImage = blend_image_mask(img.convert("RGB"),im.convert("RGB"))
        if self.transform is not None:
            prevImage = self.transform(prevImage)
            currentImage= self.transform(img)
            gt = self.transform(im)
            currentImage, gt = self.transfromFunc(currentImage, gt)

        else:

            currentImage, gt = self.transfromFunc(img, im)

        toTensor = transforms.ToTensor()
        tPrevImage = toTensor(prevImage)
        tcurrentImage  = toTensor(currentImage)
        tGt = toTensor(gt)
        return tPrevImage, tcurrentImage, tGt

    def transfromFunc(self, image, mask):
        """width, height = image.size

        angle = transforms.RandomRotation.get_params(
                                                    degrees=(0, 359))

        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        Pwidth, Pheight = transforms.RandomPerspective.get_params(
                                                                  width, 
                                                                  height,    
                                                                  0.6)

        image= TF.perspective(image, Pwidth, Pheight)
        mask= TF.perspective(mask, Pwidth, Pheight)


        degrees, translate, scale_ranges, shears = transforms.RandomAffine.get_params(  
                                                                            degrees=(30, 70), 
                                                                            translate=(0.1, 0.3), 
                                                                            scale_ranges=(0.5, 0.75),
                                                                            shears=None,
                                                                            img_size = (width, height))
        
        image = TF.affine(image, degrees, translate, scale_ranges, shears)
        mask = TF.affine(mask, degrees, translate, scale_ranges, shears)
        """

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
       
        return image, mask


if __name__ == '__main__':
    x = Coco2017Dataset()
    for prevImage, currImg, gt in x:
        print(2)

    

    """
    # Transformace pro změnu velikosti obrázku na 256x256
    transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
    ])

    # Transformace pro převod masky na tensor
    target_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = davis2017Dataset(transform=transform, target_transform=target_transform)
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for prevImage, currImg, gt in dataloader:
        print(prevImage)
    """
    