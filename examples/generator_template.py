# -*- coding: utf-8 -*-
import os
import torch
import random
import numpy as np

# reading csv files
import pandas as pd
import random

# 7 classes
LABELS = ['NONE', 'TWC', 'LVHV', 'STTC', 'RBBB', 'ALS', '1AVB']

class Generator(object):
    def __init__(self, idir, labels, batch_size = 8, val_p = 0.1):
        # a list of datums and their y values
        # split the list into training and validation
        # saving some small percentage (%10 maybe) for validation
        
        # you'll either want to balance batches (i.e. have the same number of each class in them)
        # or pick a weighted loss function s.t. you're giving less weight to classes that are more present
        self.batch_size = batch_size
        
        # idir : input directory of csvs
        self.ifiles = [os.path.join(idir, u) for u in os.listdir(idir) if u.split('.')[-1] == 'csv']
        self.labels = pd.read_csv(labels)
        
        fnames = list(self.labels['FileName'])
        self.labels = self.labels.set_index('FileName')
        
        fnames = [u for u in fnames if self.labels['Beat'][u] in LABELS]
        
        # x and y (file name for CSV, string label)
        self.ifiles = [u for u in self.ifiles if u.split('/')[-1].split('.')[0] in fnames]
        self.labels = [self.labels['Beat'][u.split('/')[-1].split('.')[0]] for u in self.ifiles]
        
        counts = []
        self.ifiles_val = []
        self.labels_val = []        
        
        for lab in LABELS:
            counts.append(self.labels.count(lab))
            
            ifiles_ = [self.ifiles[k] for k in range(len(self.ifiles)) if self.labels[k] == lab]
            N = int(counts[-1] * val_p)
            
            random.shuffle(ifiles_)
            
            self.ifiles_val.extend(ifiles_[:N])
            self.labels_val.extend([lab for u in range(N)])
            
        self.ifiles = [u for u in self.ifiles if u not in self.ifiles_val]
        self.labels = [self.labels['Beat'][u.split('/')[-1].split('.')[0]] for u in self.ifiles]
        
        self.ix = 0
        self.val_ix = 0
        
        self.train_len = len(self.ifiles) // self.batch_size
        self.val_len = len(self.ifiles_val) // self.batch_size
        
        self.on_epoch_end()
        
        return
    
    def get_batch(self):
        # return x, y as torch Tensors
        # x and y should have self.batch_size random entries in them
        x = [self.ifiles[u] for u in range(self.ix, self.ix + self.batch_size)]
        y = [LABELS.index(self.labels[u]) for u in range(self.ix, self.ix + self.batch_size)]
        
        self.ix += self.batch_size
        
        x = np.array([np.loadtxt(u, delimiter = ',') for u in x])
        # have shape (8, 5000, 12)
        # x[0, 1, 2]
        # shape (5000, 12)
        # x[0, 0], x[:,0], x[::-1,0] (5000,), x[:,[0, 1, 7]] (5000, 3)
        x = x.transpose(0, 2, 1) # (8, 12, 5000)
        
        # x is floating point and y is the index of the class it belongs to (0, 1, 2, ...)
        return torch.FloatTensor(x), torch.LongTensor(np.array(y, dtype = np.int32))
    
    def get_val_batch(self):
        # return x, y as torch Tensors
        # x and y should have self.batch_size random entries in them
        x = [self.ifiles_val[u] for u in range(self.ix_val, self.ix_val + self.batch_size)]
        y = [LABELS.index(self.labels_val[u]) for u in range(self.ix_val, self.ix_val + self.batch_size)]
        
        self.ix_val += self.batch_size
        
        x = np.array([np.loadtxt(u, delimiter = ',') for u in x])
        # have shape (8, 5000, 12)
        # x[0, 1, 2]
        # shape (5000, 12)
        # x[0, 0], x[:,0], x[::-1,0] (5000,), x[:,[0, 1, 7]] (5000, 3)
        x = x.transpose(0, 2, 1) # (8, 12, 5000)
        
        # x is floating point and y is the index of the class it belongs to (0, 1, 2, ...)
        return torch.FloatTensor(x), torch.LongTensor(np.array(y, dtype = np.int32))
    
    def on_epoch_end(self):
        # shuffle the training data
        # not necessary for validation data
        ix = list(range(len(self.ifiles)))
        random.shuffle(ix)
        
        self.ifiles = [self.ifiles[u] for u in ix]
        self.labels = [self.labels[u] for u in ix]
        
        return
    
if __name__ == '__main__':
    gen = Generator('data/ECGDataDenoised', 'data/Diagnostics.csv')