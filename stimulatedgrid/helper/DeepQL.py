import time # Evaluate frugality
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv # Imposed Environment
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Deep Q-learning using pytorch
class DeepQL(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, nb_actions):
        super(DeepQL, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.nb_actions = nb_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.nb_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
        if torch.cuda.is_available():       
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        
        return actions
    