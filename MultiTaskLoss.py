#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 22:55:21 2023

@author: nabil
"""


import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(MultiTaskLoss, self).__init__()
        self.csloss= nn.CrossEntropyLoss()
        # Define any additional parameters or initialization steps here
        # For example, you might want to pass a weight parameter to the loss function
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, predictions, targets):
        # Implement your custom loss computation here
        # Ensure that the output is a scalar tensor representing the loss

        loss1 = self.csloss(predictions[:,0,:] , targets[:,0])
        loss2 = self.csloss(predictions[:,1,:] , targets[:,1])
        loss =0.5*loss1 + 0.5*loss2

        # Apply any additional custom computations if needed
        # ...

        return loss
