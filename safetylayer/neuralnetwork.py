import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim,128)
        # self.layer2 = nn.Linear(128,64)
        self.layer3 = nn.Linear(128,output_dim)
        self.act = nn.ReLU()

    def forward(self,x):
        # torch.autograd.set_detect_anomaly(True)
        x = self.act(self.layer1(x))
        # out = self.act(self.layer2(x))
        out = self.layer3(x)
        return out