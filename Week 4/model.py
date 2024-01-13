# Scroll down for instructions

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Union

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        label = torch.tensor(label, dtype=torch.float32)
        return sample, label

'''
You have to make a DQN model which approximates the value of Q(s, a) given s & a

INSTRUCTIONS:
You must be familiar with how we make a class for any neural network, the class given below has the same format and you have to implement a neural network using linear layers and ReLU activations which takes the states and action as input and gives Q(s, a) as output,
- You have to make a DQN class which accepts appropriate input and gives appropriate output and has a train functionality
- The constructor '__init__' should initialize all the layers for the neural network as class variables
- The forward() function will accept any numpy array or torch tensor of appropriate size and output the value obtained after passing it through all layers
- The train_instance() function accepts a training dataset which is of CustomDataset datatype and represents the replay buffer, and these will be used for training the model for some fixed epochs (iterations). You should use the imported DataLoader class to sample random batches from the given training dataset while model training.
- The save_model() function has already been implemented and you should use it to save your model
- You should declare the layers of the Neural Net as class variables (Allows us to use optimizer easily)

Now some recommendaions:
- PLEASE DO USE optimizer.x
- You should declare the optimizer and the loss function in the constructor (__init__) itself as class variables, the optimizer takes parameters and learning rate as input, passing self.parameters() as params to the optimizer passes all the layers in the class (which are declared as class variables) as input by default. Adam optimizer is prefered for optimizer and Huberloss is prefered for loss function in this case. For ex:
    self.optimizer = optim.Adam(params = self.parameters(), lr = 0.001)
    self.loss_func = nn.HuberLoss(delta = 1)
- There can be a lots of different architectures for the neural net layers (it generally doesn't matter much for cartpole as it is a much easy-to-solve environment), but here is one which I've used (not the best, but it works):
     Input Size  |  Layer1 size  |  Layer2 size  |  Layer3 size  |  Output Layer Size
         5             5x64           64x16           16x8           8x1 (Of Course!)

'''

class DQN(nn.Module):

    def __init__(self,input_dim=5,output_dim=1,hidden_dims=(64,16,8),activation_fc=nn.ReLU()) -> None:
        super(DQN, self).__init__()
        self.activation_fc = activation_fc
        self.l1 = nn.Linear(input_dim, hidden_dims[0]).float()
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1]).float()
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_dims[1], hidden_dims[2]).float()
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(hidden_dims[2], output_dim).float()
        self.optimizer = optim.Adam(params=self.parameters(), lr=0.001)
        self.loss_func = nn.HuberLoss(delta=1)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.l1(x.float()))
        out = self.relu2(self.l2(out))
        out = self.relu3(self.l3(out))
        q_value = self.l4(out)
        q_value = q_value.float().view(-1,1)
        return q_value
    def to_float32(self):
        for param in self.parameters():
            if param.data.dtype == torch.float64:
                param.data = param.data.float()
            param.grad = None
    def train_instance(self, train_dataset : CustomDataset) -> None:
        train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)

        for epoch in range(15):
            total_loss = 0
            for data, labels in train_loader:
                self.to_float32()
                self.optimizer.zero_grad()
                data=data.float()
                labels = labels.float()
                q_values = self.forward(data)
                labels = labels.unsqueeze(1)

                #print(q_values.size(),labels.size())
                loss = self.loss_func(q_values, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

    def save_model(self, model_name : str):
        torch.save(self.state_dict(), model_name) # Saves the model with the value of model_name as its name


