# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import logging
import random

### DESCRIPTION:
### Loads a random X variable from the downloaded and unzipped folder and plots it

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "data/ECGDataDenoised", help = "path to directory of CSV files (X) of 12-lead ECG signals")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()
    
    # os.listdir gets a list of files from args.idir
    # for each of those we join the path the of input directory for the full path
    ifiles = [os.path.join(args.idir, u) for u in os.listdir(args.idir)]
    
    # shuffle the list and pick the first one
    random.shuffle(ifiles)
    ifile = ifiles[0]
    
    # Load the data using NumPy
    # we specify that the numbers are comma separated
    X = np.loadtxt(ifile, delimiter = ',')
    logging.info('X has shape {}...'.format(X.shape))
    
    fig, axes = plt.subplots(ncols = 2)
    
    # plot the 2d array as an image with an aspect ratio = 1
    axes[0].imshow(X, extent = (0, 1, 0, 1))
    
    axes[1].plot(X[:,0])
    axes[1].set_xlabel('time')
    axes[1].set_ylabel('voltage')
    
    plt.show()

if __name__ == '__main__':
    main()
