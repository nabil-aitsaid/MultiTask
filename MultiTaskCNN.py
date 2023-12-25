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
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=30,kernel_size=(5,5))
        self.pool1=nn.MaxPool2d(kernel_size=(5,5))
        self.conv2 = nn.Conv2d(in_channels=30,out_channels=20,kernel_size=(3,3))
        self.pool2=nn.MaxPool2d(kernel_size=(2,2))
        self.fc1=nn.Linear(in_features=20*5*5,out_features=256)

        self.fc2=nn.Linear(in_features=256,out_features=64)
        self.fc31=nn.Linear(in_features=64,out_features=10)
        self.fc32=nn.Linear(in_features=64,out_features=10)
        self.relu=nn.LeakyReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x1 = self.fc31(x)
        x2 = self.fc32(x)
        return torch.cat((x1.unsqueeze(1) , x2.unsqueeze(1)),dim=1)
