from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange



class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim,
        trainingDataloader: Dataset,
        validatinDataloader: Optional[Dataset] = None,
        epochs: int = 50,
        epoch: int = 0,
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
    
    def run(self):
        
        progressbar = trange(self.epochs, desc="Progress")
        for i in progressbar:
            self.epoch += 1 
            self._train()
            
            if self.validatinDataloader is not None:
                self._validate()


        return self.trainingLoss, self.validationLoss

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
        validLosses = []
        batch_iter = tqdm(
            enumerate(self.validatinDataloader),
            "Validation",
            total=len(self.validatinDataloader),
            leave=False,
        )        

        #for i, (input, gt) in batch_iter:
        for i, (prevImage, currImg, gt) in batch_iter:
            with torch.no_grad():
                out = self.model(prevImage, currImg)
                loss = self.criterion(out,gt)
                validLosses.append(loss.item())
            
        self.validationLoss.append(np.mean(validLosses))
        batch_iter.close()

