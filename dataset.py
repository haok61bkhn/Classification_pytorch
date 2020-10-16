from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from split_data import split_data
import os
class Dataset:
    def __init__(self,data_dir,out_dir,input_size,batch_size=8,rate=80):
        self.data_dir=data_dir
        self.out_dir=out_dir
        self.input_size=input_size
        self.rate=rate
        self.batch_size=batch_size
        self.create_folder()
        self.get_transform()
        self.get_Dataloader()


    def create_folder(self):
        split_data(self.data_dir,self.out_dir,self.rate)
    
    def get_transform(self):
       self.transforms={
            'train': transforms.Compose([
                transforms.Resize((self.input_size,self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((self.input_size,self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    def get_Dataloader(self):
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.out_dir, x), self.transforms[x]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        self.dataLoaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# X=Dataset("./data","dataset",224)