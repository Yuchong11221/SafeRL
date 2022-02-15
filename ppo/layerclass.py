import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = nn.Linear(self.input_dim,128)
        self.layer2 = nn.Linear(128,64)
        self.layer3 = nn.Linear(64,self.output_dim)
        self.act = nn.ReLU()

    def forward(self,x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        output = self.layer3(x)
        return output


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = nn.Linear(self.input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, self.output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        output = self.layer3(x)
        return output