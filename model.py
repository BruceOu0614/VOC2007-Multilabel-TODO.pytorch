import os
import time

import torch
import torch.nn.functional
from torch import nn, Tensor

import torchvision.models as models


import math

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # TODO: CODE BEGIN
        self.resnet18 = models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 20)
        #raise NotImplementedError
        # TODO: CODE END

    def forward(self, images: Tensor) -> Tensor:
        # TODO: CODE BEGIN
        # logits = xxx
        x = self.resnet18(images)
        x = self.fc(x)
        #m = nn.Softmax()
        #x = m(x)
        #print("x = ", x)
        return x
        #raise NotImplementedError
        # TODO: CODE END

    def loss(self, logits: Tensor, multilabels: Tensor) -> Tensor:
        # TODO: CODE BEGIN
        #loss_func = torch.nn.CrossEntropyLoss()
        #print("logits = ", logits)
        #print("multilables = ", multilabels)
        '''loss = 0
        labels = torch.zeros([32,1], dtype=torch.int64)
        labels = labels.cuda()
        for x in range(0,32):
            for y in range(0,20):
                #print("logits[x][y] = ", float(logits[x][y])+1e-9)
                #print("multilabels[x][y] = ", float(multilabels[x][y]))
                loss += float(multilabels[x][y])*math.log(float(logits[x][y])+1e-9)
                
                if multilabels[x][y] == 1:
                    labels[x].append = y
                #print("loss = ", loss)
        print("labels = ", labels)
        #print("hand loss = ", loss)
        #print("multilabels = ", multilabels)
        #print("torch.max(multilabels, 1)[1] = ", torch.max(multilabels, 1))
        loss = loss_func(logits, labels)
        '''
        '''criterion = nn.NLLLoss2d()
        logits = nn.functional.log_softmax(logits, dim=1)
        loss = criterion(logits, multilabels)
        print("loss = ", loss)
        '''
        #print("loss = ", loss)
        loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(logits), multilabels)
        return loss
        #loss = loss_func(logits, multilabels)
        #raise NotImplementedError
        # TODO: CODE END

    def save(self, path_to_checkpoints_dir: str, step: int) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir,
                                          'model-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str) -> 'Model':
        self.load_state_dict(torch.load(path_to_checkpoint))
        return self
