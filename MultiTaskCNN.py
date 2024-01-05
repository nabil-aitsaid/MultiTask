#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:29:44 2023

@author: nabil
"""

import torch
import torch.nn as nn
class MultiTaskCNN(nn.Module):
    def __init__(self):
        super(MultiTaskCNN,self).__init__()
        self.conv = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(5,5)) #60
        self.pool=nn.AvgPool2d(kernel_size=(6,6)) # 10

        self.relu=nn.LeakyReLU()

        self.fc1=nn.Linear(in_features=1600,out_features=256)
        self.dropout= nn.Dropout()
        self.fc21=nn.Linear(in_features=256,out_features=10)
        self.fc22=nn.Linear(in_features=256,out_features=10)
     

    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        
        x1 =self. dropout(x)
        x1 = self.fc21(x1)

        x2 =self. dropout(x)
        x2 = self.fc22(x2)
        #returned value size : (batch_size,2,10)
        return torch.cat((x1.unsqueeze(1) , x2.unsqueeze(1)),dim=1)
