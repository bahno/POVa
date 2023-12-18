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
            gt = self.transform(gt)

        """
        if self.pad_mirroring:
            img = Pad(padding=self.pad_mirroring, padding_mode="reflect")(img)"""

        return prevImage, currImg, gt


"""
class cocoDataset(Dataset):
    def __init__(self, annotationsFile, dataDir, transform=None, target_transform=None):
        annotations = []
        with open(annotationsFile, 'r') as f:
            annotations = json.load(f)
        self.imgLabels = annotations
        print(annotations)

        #self.imgLabels = pd.read_json(annotationsFile)
        self.dataDir = dataDir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgLabels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataDir, self.imgLabels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.imgLabels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label










class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, idx):
        img_id = self.coco.getImgIds()[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.root_dir}/{img_info['file_name']}"
        img = Image.open(img_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Předpokládáme, že máme binární masku pro segmentaci
        mask = self.coco.annToMask(annotations[0])

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            mask = self.target_transform(mask)

        return img, mask
"""

if __name__ == '__main__':

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
