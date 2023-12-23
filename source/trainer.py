from typing import Optional
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange


class Trainer:
    def __init__(
        self,
        model : torch.nn.Module,
        criterion : torch.nn.Module,
        optimizer : torch.optim,
        trainingDataloader : Dataset,
        validatinDataloader : Optional[Dataset] = None,
        epochs : int = 50,
        epoch : int = 0,
        logValidation:  bool = True
    ):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainingDataloader = trainingDataloader
        self.validatinDataloader = validatinDataloader
        self.epochs = epochs
        self.epoch = epoch
        self.trainingLoss = []
        self.validationLoss = []
        self.bestModel = {
            'model' : None,
            'loss' : None,
            'epoch' : 0
        }
        self.logValidation = logValidation
        self.valAccuracy = []
        self.trainAccuracy = []
        self.valDice = []
        self.trainAccuracy = []

    def run(self):
        
        progressbar = trange(self.epochs, desc="Progress")
        for i in progressbar:
            self.epoch += 1 
            self._train()
        
            if self.validatinDataloader is not None:
                self._validate()


        return self.trainingLoss, self.validationLoss, self.bestModel

    def _train(self):
        self.model.train()
        trainingLosses = []  
        batch_iter = tqdm(
            enumerate(self.trainingDataloader),
            "Training",
            total=len(self.trainingDataloader),
            leave=False,
        )

        for i, (prevImage, currImg, gt) in batch_iter:
            self.optimizer.zero_grad() #set grads to None
            out = self.model(prevImage, currImg)

            loss = self.criterion(out, gt)
            trainingLosses.append(loss.item())
 

            
            loss.backward()
            self.optimizer.step() #update grads


        self.trainingLoss.append(np.mean(trainingLosses))
        batch_iter.close()
    
    def _validate(self):
        self.model.eval()
        valLosses = []
        valDice = []
        valAccuracy = []
        batch_iter = tqdm(
            enumerate(self.validatinDataloader),
            "Validation",
            total=len(self.validatinDataloader),
            leave=False,
        )        

        loss = None
        for i, (prevImage, currImg, gt) in batch_iter:
            with torch.no_grad():
                out = self.model(prevImage, currImg)
                loss = self.criterion(out,gt)
                valLosses.append(loss.item())
                #valAccuracy.append(accuracy(out,gt))
                #valDice.append(DiceLoss(out,gt))

        if self.logValidation:
            if (self.bestModel['model'] == None):
                self.bestModel['model'] = copy.deepcopy(self.model)
                self.bestModel['loss'] = np.mean(valLosses)
                self.bestModel['epoch'] = self.epoch

            else:
                if (np.mean(valLosses) < self.bestModel['loss']):
                    self.bestModel['model'] = copy.deepcopy(self.model)
                    self.bestModel['loss'] = np.mean(valLosses)
                    self.bestModel['epoch'] = self.epoch
            

        self.validationLoss.append(np.mean(valLosses))
        print(f"Epoch: {self.epochs}  Loss: {self.validationLoss[-1]}")
        batch_iter.close()

