from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader    
import numpy as np
import pandas as pd
from PIL import Image
import torch
import os

class DavisEvalDataloader(Dataset):
    def __init__(self, path, FolderName, transform = None):
        self.pathAnnotations = path + 'Annotations/480p/'  + FolderName
        self.pathImages      = path + 'JPEGImages/480p/' + FolderName
        self.pathMerged      = path + 'Merged/480p/' + FolderName
        self.transform       = transform
        self.NumberofFrame   =  len(os.listdir(self.pathAnnotations))
        self.AnnotationsCurr = [str(file).zfill(5) + '.jpg' for file in range(1,self.NumberofFrame )]
        self.AnnotationsMerged= [str(file).zfill(5) + '.jpg' for file in range(self.NumberofFrame - 1)]

        self.AnnotationsMasks= [str(file).zfill(5) + '.png' for file in range(1,self.NumberofFrame)]

    def __len__(self):
        len(self.files)

    def __getitem__(self, idx):
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

        gt = Image.fromarray(np.uint8((gt) * 255))
        if  self.transform is not None:
            currImg = self.transform(currImg)
            prevImage = self.transform(prevImage)
            gt = self.target_transform(gt)


        return prevImage, currImg, gt


    



class evaluation():
    def __init__(self,
                model : nn.Module = None, 
                dataloader : Dataset = None):
        self.model = model,
        self.dataloader = dataloader
        self.statsFrame= pd.DataFrame({'FileName' : [], 'f1Score' : [], 'Jaccard' : [] })

    def _computePrecisionScore(y, pred):
        return precision_score(y, pred)
    
    def _computeRecallScore(y, pred):
        return recall_score(y, pred)
    
    def _f1Score(y, pred):
        return f1_score(y, pred)

    def _compute_jaccard_similarity_score(y, pred):
        return len(set(y).intersection(set(pred))) / float(len(set(y).union(set(pred))))
    

    def evalDavis(self):

       

        path = "../datasets/Davis/train480p/DAVIS/"
        AnnotationsFile =  pd.read_csv(
            path + "ImageSets/2017/val.txt",
            header = None,
            names = ["ImageDirNames"])
        
    
        for idx in range(0,len(AnnotationsFile)):
            FileName = AnnotationsFile.at[idx, "ImageDirNames"]
            DavisDataloader = DavisEvalDataloader(path, FileName)
            jaccardArray = []
            f1ScoreArray = []
            firstIteration = 1
            NewMergedImage = None
            for prevImage, current, mask in DavisDataloader:
                if (firstIteration == 1):
                    firstIteration = 0
                    with torch.no_grad():
                        out = self.model(prevImage, current)
                        # TODO prahování masky, aby měla stejný tvar jako maska z Dataloaderu
                        jaccardArray.append(self._compute_jaccard_similarity_score(out,mask))
                        f1ScoreArray.append(self._f1Score(out,mask))
                else:
                    with torch.no_grad():
                        out = self.model(NewMergedImage, current)
                        # TODO prahování masky, aby měla stejný tvar jako maska z Dataloaderu

                        jaccardArray.append(self._compute_jaccard_similarity_score(out,mask))
                        f1ScoreArray.append(self._f1Score(out,mask))
                NewMergedImage # TODO z out masky vytvořit 

            self.statsFrame = self.statsFrame.append(FileName, np.mean(f1ScoreArray), np.mean(jaccardArray))
            

        return self.statsFrame["f1Score"].mean(), self.statsFrame["Jaccard"].mean()






