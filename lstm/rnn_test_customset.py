# Import 
import torch
import torchvision  # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For a nice progress bar!
from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import numpy as np

# Custom Dataset: returns input sequences and labels.
class MyDataset(Dataset):
    def __init__(self, input, seq_len):
        self.input = input
        self.seq_len = seq_len
    def __getitem__(self, item):
        return input[item: item + self.seq_len], input[item + self.seq_len]
    def __len__(self):
        return len(self.input) - self.seq_len


# input = np.arange(1,8) # output [1 2 3 4 5 6 7]
input = np.arange(1,8).reshape(-1, 1) #output [[1] [2] [3] [4] [5] [6] [7]], shape(7,1)
input = torch.tensor(input, dtype=torch.float)

dataset = MyDataset(input, 3)
dataloader = DataLoader(dataset, batch_size=2)

for input_sequence, label in dataloader:
    print(input_sequence.numpy())
    print(label)

# Input type 2: of shape (Batch Size, Sequence Length, Input Dimension)
# Input type 1: inp.permute(1,0,2) # switch dimension 0 and dimension 1

class RNNModel(torch.nn.Module):
    def __init__(self, input_size, HL_size):
        super(RNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size=input_size,
                                hidden_size=Hidden Size(HS),
                                num_layers=number of stacked RNN,
                                bidirectional=True/False,
                                batch_first=True default is False)
        # If you want to use output for next layer then
        # self.linear2 = torch.nn.Linear(#Direction * HS , Output_size)
        # If you want to use hidden for next layer then
        self.linear2 = torch.nn.Linear(HS , Output_size)

    
    def forward(self, input):
        out, hidden_ = self.rnn(input)
        #out: Select which time step data you want for linear layer 
        out = self.linear2(out)

        out = self.linear2(hidden_)