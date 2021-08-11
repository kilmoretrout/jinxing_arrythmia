# jinxing_arrythmia
Github for student and I to contribute to for his project

I've decided we'll be using the data from this paper:
https://www.nature.com/articles/s41597-020-0386-x

## Downloading the data
Download the denoised ECG data.
https://figshare.com/articles/dataset/ECGDataDenoised_zip/8378291?backTo=/collections/ChapmanECG/4560497

Download the labels:
https://figshare.com/articles/dataset/Diagnostics_xlsx/8360408?backTo=/collections/ChapmanECG/4560497

and their names:
https://figshare.com/articles/dataset/ConditionNames_xlsx/8360411?backTo=/collections/ChapmanECG/4560497

## Goals:
### Week 1 / 3:
- [x] Make a balanced sampler of the downloaded data for training.
- [ ] Write training routine from example scripts given. Tutorials below.

### Goals for Friday (8/13):
In line 51 of examples/training_template.py:
```
### instantiate your model here ###
# model = MyModel()
```
- [ ] Write a class like (at the top of training_template.py):
```
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
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
```

But replace self.conv1 and 2 with nn.Conv1d() and choose a kernel size etc. that fit our data, see (https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) and (https://towardsdatascience.com/understanding-1d-and-3d-convolution-neural-network-keras-9d8f76e29610) as reference, that latter being a conceptual reference.

- [ ] Choose a loss function:

```
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, SmoothL1Loss, BCELoss, BCEWithLogitsLoss
```

Look at https://pytorch.org/docs/stable/nn.html#loss-functions for descriptions and the appropriate output function to use.  Hint: a good choice might be ```nn.NLLLoss()```, while adding:

```
def __init__(self):
    ...
    self.fc3 = Linear(..., 7) # we have 7 classes
    self.out = nn.LogSoftmax(dim = -1)

def forward(x, self):
    ...
    return self.out(x)
```

- [ ] Add an arugment for the input directory and label CSV to the script and instantiate the generator class in lin 60.
- [ ] Attempt to run the script with a small batch size on your local CPU.

Tutorials: 

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

https://pytorch.org/docs/stable

