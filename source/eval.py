import PIL
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import torch
import os

from torchvision.transforms import transforms

from data_augmentation import DataAugmenter
from source.video_segmentation_model import ourModel


class DavisEvalDataloader(Dataset):
    def __init__(self, path, FolderName, transform=None, target_transform=None):
        self.pathAnnotations = path + 'Annotations/480p/' + FolderName
        self.pathImages = path + 'JPEGImages/480p/' + FolderName
        self.pathMerged = path + 'Merged/480p/' + FolderName
        self.transform = transform
        self.target_transform = target_transform
        self.NumberofFrame = len(os.listdir(self.pathAnnotations))
        self.AnnotationsCurr = [str(file).zfill(5) + '.jpg' for file in range(2, self.NumberofFrame)]
        self.AnnotationsMerged = [str(file).zfill(5) + '.jpg' for file in range(1, self.NumberofFrame - 1)]

        self.AnnotationsMasks = [str(file).zfill(5) + '.png' for file in range(2, self.NumberofFrame)]

    def __len__(self):
        len(self.files)

    def __getitem__(self, idx) -> (Tensor, Tensor, Tensor):
        currImg = Image.open(
            os.path.join(self.pathImages, self.AnnotationsCurr[idx]).replace(os.sep,
                                                                             '/')).convert(
            "RGB")
        prevImage = Image.open(
            os.path.join(self.pathMerged, self.AnnotationsMerged[idx]).replace(os.sep,
                                                                               '/')).convert(
            "RGB")
        gt = np.array(Image.open(
            os.path.join(self.pathAnnotations, self.AnnotationsMasks[idx]).replace(os.sep,
                                                                                   '/')).convert(
            "L"),
            dtype=np.float32)

        gt = ((gt / np.max([gt.max(), 1e-8])) > 0.5).astype(np.float32)

        gt = Image.fromarray(np.uint8(gt * 255))

        if self.transform is not None:
            currImg = self.transform(currImg)
            prevImage = self.transform(prevImage)
            gt = self.target_transform(gt)

        return prevImage, currImg, gt


class evaluation:
    def __init__(self, model: ourModel = None, dataset: Dataset = None):
        self.model: ourModel = model
        self.dataset = dataset
        self.statsFrame = pd.DataFrame({'FileName': [], 'f1Score': [], 'Jaccard': []})

    @staticmethod
    def _computePrecisionScore(y, pred):
        return precision_score(y, pred)

    @staticmethod
    def _computeRecallScore(y, pred):
        return recall_score(y, pred)

    @staticmethod
    def _f1Score(y, pred):
        _f1score = f1_score(y, pred)
        return _f1score

    @staticmethod
    def _compute_jaccard_similarity_score(y, pred):
        y_flat = y.flatten()
        pred_flat = pred.flatten()
        return len(set(y_flat).intersection(set(pred_flat))) / float(len(set(y_flat).union(set(pred_flat))))

    def evalDavis(self):
        custom_transforms = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        target_transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.ToTensor(),

        ])

        path = "../datasets/Davis/train480p/DAVIS/"
        AnnotationsFile = pd.read_csv(
            path + "ImageSets/2017/val.txt",
            header=None,
            names=["ImageDirNames"])

        for idx in range(len(AnnotationsFile)):
            FileName = AnnotationsFile.at[idx, "ImageDirNames"]
            DavisDataloader = DavisEvalDataloader(path, FileName, transform=custom_transforms,
                                                  target_transform=target_transform)
            jaccardArray = []
            f1ScoreArray = []
            firstIteration = True
            data_augmenter = DataAugmenter()
            threshold = -3.0

            for prevImage, current, mask in DavisDataloader:
                if firstIteration:
                    firstIteration = False
                    with torch.no_grad():
                        out = self.model.forward(
                            prevImage.unsqueeze(0),
                            current.unsqueeze(0)
                        )

                        # TODO hodnota thresholdu je nyní nastavena na -1.5,
                        #  je potřeba zjistit, jaký threshold je nejlepší
                        out_threshold = torch.where(out > threshold, 0.0, 1.0)
                        """
                        plt.imshow(out_threshold.squeeze(), cmap='viridis', interpolation='nearest')
                        plt.colorbar()
                        plt.show()

                        plt.imshow(mask.squeeze(), cmap='viridis', interpolation='nearest')
                        plt.colorbar()
                        plt.show()
                        """

                        jaccardArray.append(
                            self._compute_jaccard_similarity_score(out_threshold.squeeze(), mask.squeeze()))
                        f1ScoreArray.append(
                            self._f1Score(
                                out_threshold.squeeze().numpy().flatten() == 1,
                                mask.squeeze().numpy().flatten() == 1
                            )
                        )
                else:
                    with torch.no_grad():
                        if data_augmenter.prev_merged is None:
                            print("merged_current is None")
                        out = self.model.forward(
                            transforms.ToTensor()(data_augmenter.prev_merged).unsqueeze(0),
                            current.unsqueeze(0)
                        )
                        # TODO hodnota thresholdu je nyní nastavena na -1.5,
                        #  je potřeba zjistit, jaký threshold je nejlepší

                        out_threshold = torch.where(out > threshold, 1.0, 0.0)
                        jaccardArray.append(
                            self._compute_jaccard_similarity_score(out_threshold.squeeze(), mask.squeeze()))
                        f1ScoreArray.append(self._f1Score(
                            out_threshold.squeeze().numpy().flatten() == 1,
                            mask.squeeze().numpy().flatten() == 1
                        ))

                data_augmenter.set_prev_images(
                    transforms.ToPILImage()(current),
                    transforms.ToPILImage()(out_threshold[0]))
                data_augmenter.merge_image_and_mask()

            new_row = {'FileName': FileName, 'f1Score': np.mean(f1ScoreArray), 'Jaccard': np.mean(jaccardArray)}
            print(f"idx: {idx}\ndata: {new_row}")
            self.statsFrame = self.statsFrame._append(new_row, ignore_index=True)

        return self.statsFrame["f1Score"].mean(), self.statsFrame["Jaccard"].mean()
