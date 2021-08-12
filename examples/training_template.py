# -*- coding: utf-8 -*-
import os
import logging
import argparse

from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, SmoothL1Loss, BCELoss, BCEWithLogitsLoss
import torch

from torch import nn

import torch.optim as optim
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--weights", default = "None")
    
    parser.add_argument("--n_epochs", default = "100", help = "number of times to go through the training set")
    parser.add_argument("--lr", default = "0.001", help = "learning rate for Adam optimizer")
    
    parser.add_argument("--odir", default = "training_output")
    parser.add_argument("--tag", default = "test")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.debug('root: made output directory {0}'.format(args.odir))
    # ${odir_del_block}

    return args

def main():
    args = parse_args()

    # do we have a GPU?
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    ### instantiate your model here ###
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(16, 33, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


net = Net()
    # model = MyModel()
    model = model.to(device)
    
    # do we have weights?
    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location = device)
        model.load_state_dict(checkpoint)

    ### Define your data generator class here
    ## generator = MyGenerator(*args)
    ## val_generator = MyValGenerator(*args)

    ### Define your criterion (loss) here
    ## criterion = nn.NLLLoss() # negative log-likelihood seems like a good choice
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    min_val_loss = np.inf
    # count to stop early if validation loss isn't going down
    early_count = 0

    # save some metrics to look at later...
    history = dict()
    history['epoch'] = []
    history['loss'] = []
    history['val_loss'] = []
    # ${define_extra_history}

    print('training...')
    for ix in range(int(args.n_epochs)):
        model.train()

        losses = []
        accuracies = []

        for ij in range(len(generator)):
            optimizer.zero_grad()
            x, y = generator.get_batch()

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # compute accuracy in CPU with sklearn
            y_pred = np.exp(y_pred.detach().cpu().numpy())
            y = y.detach().cpu().numpy()

            y_pred = np.argmax(y_pred, axis=1)

            # append metrics for this epoch
            accuracies.append(accuracy_score(y, y_pred))

            # print progress to console
            # loss should be going down...
            if (ij + 1) % 10 == 0:
                logging.info(
                    'root: Epoch {0}, step {3}: got loss of {1}, acc: {2}'.format(ix, np.mean(losses),
                                                                                  np.mean(accuracies), ij + 1))
        ## VERY IMPORTANT if your model has stochastic regularization like Dropout
        ## or batch normalization, etc.
        model.eval()

        val_losses = []
        val_accs = []
        for step in range(len(val_generator)):
            with torch.no_grad():
                x, y = val_generator.get_batch()

                x = x.to(device)
                y = y.to(device)
    
                loss = criterion(y_pred, y)
                # compute accuracy in CPU with sklearn
                y_pred = np.exp(y_pred.detach().cpu().numpy())
                y = y.detach().cpu().numpy()

                y_pred = np.argmax(y_pred, axis=1)

                # append metrics for this epoch
                val_accs.append(accuracy_score(y, y_pred))
                val_losses.append(loss.detach().item())

        val_loss = np.mean(val_losses)

        logging.info(
            'root: Epoch {0}, got val loss of {1}, acc: {2} '.format(ix, val_loss, np.mean(val_accs)))

        history['epoch'].append(ix)
        history['loss'].append(np.mean(losses))
        history['val_loss'].append(np.mean(val_losses))
        # ${save_extra_history}

        val_loss = np.mean(val_losses)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print('saving weights...')
            torch.save(model.state_dict(), os.path.join(args.odir, '{0}.weights'.format(args.tag)))

            early_count = 0
        else:
            early_count += 1

            # early stop criteria
            if early_count > int(args.n_early):
                break
        
        ## used to shuffle the data and let the generator know were iterating through it again
        generator.on_epoch_end()

        # save the training history using Pandas (CSV library)
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(args.odir, '{}_history.csv'.format(args.tag)), index = False)
