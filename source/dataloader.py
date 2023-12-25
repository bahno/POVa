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
from torchvision.transforms import v2
import random
import torchvision.transforms.functional as TF
from PIL import ImageFile
from itertools import product

ImageFile.LOAD_TRUNCATED_IMAGES = True



class davis2017Datasetv2(Dataset):
    def __init__(self,
                 dataDir: str = '../datasets/Davis/train480p/DAVIS/',
                 annotationsFile: str = '../datasets/Davis/train480p/DAVIS/ImageSets/2017/train.txt',
                 transform: transforms = None,
                 target_transform: transforms = None,
                 train : bool = True
                 ):
        self.dataDir = dataDir
        fileName = pd.read_csv(annotationsFile, header=None, names=["ImageDirNames"])
        self.transform = transform
        self.target_transform = target_transform
        self.CurrImages = []
        self.MasksImages = []
        self.PrevImages = []
        if train:
            self.allSeqFromDirectoryTrain(dataDir, fileName['ImageDirNames'])
        else:
            self.allSeqFromDirectoryVal(dataDir, fileName['ImageDirNames'])


    def __len__(self):
        return len(self.CurrImages)

    def __getitem__(self, idx):
        currImg = Image.open(self.CurrImages[idx].replace(os.sep,
                                                                                                           '/')).convert(
            "RGB")
        prevImage = Image.open(
            os.path.join(self.PrevImages[idx]).replace(
                os.sep,
                '/')).convert(
            "RGB")
        gt = np.array(Image.open(
            os.path.join(self.MasksImages[idx]).replace(os.sep,
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
    
    def allSeqFromDirectoryTrain(self, path, names):
        
        for n in names: 
            NumberofFrame = len(os.listdir(path + 'JPEGImages/480p/' + n))

            AnnotationsCurr = [path + 'JPEGImages/480p/' + n + '/' + str(file).zfill(5) + '.jpg' for file in range(2, NumberofFrame)]
            AnnotationsCurrAugmented = [path + 'AugmentedJPEGImages/480p/' + n + '/' + str(file).zfill(5) + '.jpg' for file in range(2, NumberofFrame)]

            AnnotationsMerged = [path + 'Merged/480p/' + n + '/' + str(file).zfill(5) + '.jpg' for file in range(1,NumberofFrame - 1)]
            AnnotationsMergedAugment = [path + 'AugmentedMerged/480p/' + n + '/' + str(file).zfill(5) + '.jpg' for file in range(1,NumberofFrame - 1)]
            AnnotationsMasks = [path + 'Annotations/480p/' + n + '/' + str(file).zfill(5) + '.png' for file in range(2, NumberofFrame)]
            AnnotationsMasksAugment = [path + 'AugmentedAnnotations/480p/' + n + '/' + str(file).zfill(5) + '.png' for file in range(2, NumberofFrame)]

            self.CurrImages  += AnnotationsCurr + AnnotationsCurrAugmented
            self.PrevImages += AnnotationsMergedAugment + AnnotationsMerged
            self.MasksImages += AnnotationsMasks + AnnotationsMasksAugment

    def allSeqFromDirectoryVal(self, path, names):
        for n in names: 
            NumberofFrame = len(os.listdir(path + 'JPEGImages/480p/' + n))

            AnnotationsCurr = [path + 'JPEGImages/480p/' + n + '/' + str(file).zfill(5) + '.jpg' for file in range(2, NumberofFrame)]
            AnnotationsMerged = [path + 'Merged/480p/' + n + '/' + str(file).zfill(5) + '.jpg' for file in range(1,NumberofFrame - 1)]
            AnnotationsMasks = [path + 'Annotations/480p/' + n + '/' + str(file).zfill(5) + '.png' for file in range(2, NumberofFrame)]

            self.CurrImages  += AnnotationsCurr
            self.PrevImages += AnnotationsMerged
            self.MasksImages += AnnotationsMasks
    
        

























class davis2017Dataset(Dataset):
    def __init__(self,
                 dataDir: str = '../datasets/Davis/train480p/DAVIS/',
                 annotationsFile: str = '../datasets/Davis/train480p/DAVIS/ImageSets/2017/train.txt',
                 seqNum: int = 0,
                 transform: transforms = None,
                 target_transform: transforms = None,
                 train: bool = True,
                 ):
        self.dataDir = dataDir

        self.train = train
        data = {'ImageDirNames': []}
        fileName = pd.read_csv(annotationsFile, header=None, names=["ImageDirNames"])
        if (train):

            dirs = ["AugmentedJPEGImages/480p/", "JPEGImages/480p/"]
            pom = pd.concat(
                [pd.DataFrame({'path': [path] * len(fileName), 'ImageDirNames': fileName['ImageDirNames']}) for path in
                 dirs],
                ignore_index=True
            )

            self.imgDir = pd.DataFrame(data)

            self.imgDir["ImageDirNames"] = pom['path'] + pom['ImageDirNames']

            self.imgDirMerge = pd.DataFrame(data)
            dirs = ["AugmentedMerged/480p/", "Merged/480p/"]
            pom = pd.concat(
                [pd.DataFrame({'path': [path] * len(fileName), 'ImageDirNames': fileName['ImageDirNames']}) for path in
                 dirs],
                ignore_index=True
            )
            self.imgDirMerge["ImageDirNames"] = pom['path'] + pom['ImageDirNames']

            dirs = ["AugmentedAnnotations/480p/", "Annotations/480p/"]
            pom = pd.concat(
                [pd.DataFrame({'path': [path] * len(fileName), 'ImageDirNames': fileName['ImageDirNames']}) for path in
                 dirs],
                ignore_index=True
            )
            self.gtImgDir = pd.DataFrame(data)
            self.gtImgDir["ImageDirNames"] = pom['path'] + pom['ImageDirNames']

            # pom['path'].replace({"AugmentedJPEGImages/480p/" : "AugmentedAnnotations/480p/",
            #                                       "MergedImages/480p/" : "Annotations/480p/"}) + pom['ImageDirNames']


        else:
            dirs = ["JPEGImages/480p/"]
            pom = pd.concat(
                [pd.DataFrame({'path': [path] * len(fileName), 'ImageDirNames': fileName['ImageDirNames']}) for path in
                 dirs],
                ignore_index=True
            )

            self.imgDir = pd.DataFrame(data)
            dirs = ["Annotations/480p/"]
            self.imgDir["ImageDirNames"] = pom['path'] + pom['ImageDirNames']

            self.gtImgDir = pd.DataFrame(data)
            self.gtImgDir["ImageDirNames"] = pom['path'].replace({
                "JPEGImages/480p/": "Annotations/480p/"}) + pom['ImageDirNames']

            """
            dirs = ["JPEGImages/480p/"]
            pom = pd.concat(
                    [pd.DataFrame({'path': [path] * len(fileName), 'ImageDirNames': fileName['ImageDirNames']}) for path in dirs], 
                                   ignore_index=True
                                )
            self.imgDir = pd.DataFrame(data)
            self.imgDir["ImageDirNames"] = pom['path'] + pom['ImageDirNames']

            self.gtImgDir = pd.DataFrame(data)
            self.gtImgDir = pom['path'].replace({"JPEGImages/480p/" : "Annotations/480p/"}) + pom['ImageDirNames']"""
            print(self.gtImgDir)

        self.transform = transform
        self.target_transform = target_transform
        self.seqNum = str(seqNum).zfill(5)
        self.nextSeqNum = str(seqNum + 1).zfill(5)

    def __len__(self):
        return len(self.imgDir)

    def __getitem__(self, idx):
        currImg = Image.open(
            os.path.join(self.dataDir, self.imgDir.at[idx, "ImageDirNames"], self.seqNum + '.jpg').replace(os.sep,
                                                                                                           '/')).convert(
            "RGB")
        prevImage = Image.open(
            os.path.join(self.dataDir, self.imgDirMerge.at[idx, "ImageDirNames"], self.nextSeqNum + '.jpg').replace(
                os.sep,
                '/')).convert(
            "RGB")
        gt = np.array(Image.open(
            os.path.join(self.dataDir, self.gtImgDir.at[idx, "ImageDirNames"], self.nextSeqNum + '.png').replace(os.sep,
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
                 datasetDir='../datasets/coco2017/train/data/',
                 annotationsFile='../datasets/coco2017/raw/instances_train2017.json',
                 transform=None):
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
        prevImage = blend_image_mask(img.convert("RGB"), im.convert("RGB"))
        if self.transform is not None:
            prevImage = self.transform(prevImage)
            currentImage = self.transform(img)
            gt = self.transform(im)
            currentImage, gt = self.transfromFunc(currentImage, gt)

        else:

            currentImage, gt = self.transfromFunc(img, im)

        toTensor = transforms.ToTensor()
        tPrevImage = toTensor(prevImage)
        tcurrentImage = toTensor(currentImage)
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
    # x = Coco2017Dataset()
    # for prevImage, currImg, gt in x:
    #    print(2)

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

    dataset = davis2017Datasetv2(transform=transform, target_transform=target_transform)
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for prevImage, currImg, gt in dataloader:
        
        print(prevImage)
        break
