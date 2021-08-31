# -*- coding: utf-8 -*-
import os
import torch
import random
import numpy as np

class Generator(object):
    def __init__(self, idir, batch_size = 8):
        # a list of datums and their y values
        # split the list into training and validation
        # saving some small percentage (%10 maybe) for validation
        # you'll either want to balance batches (i.e. have the same number of each class in them)
        # or pick a weighted loss function s.t. you're giving less weight to classes that are more present
        self.batch_size = batch_size
        
        return
    
    def get_batch(self):
        # return x, y as torch Tensors
        # x and y should have self.batch_size random entries in them 
        # x is floating point and y is the index of the class it belongs to (0, 1, 2, ...)
        return torch.FloatTensor(np.array(x)), torch.LongTensor(np.array(y, dtype = np.int32))
    
    def on_epoch_end(self):
        # shuffle the training data
        # not necessary for validation data
        return
