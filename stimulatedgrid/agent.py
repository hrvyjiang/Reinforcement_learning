import time # Evaluate frugality
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv # Imposed Environment
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from helper import DeepQL

# Training of the agent
class RL_Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, nb_actions,
                max_mem_size = 100000, eps_end = 0.01, eps_dec = 5e-4):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(nb_actions)]
        self.mem_size = max_mem_size
        self.mem_counter = 0
        
        self.Q_eval = DeepQL(self.lr, nb_actions=nb_actions, input_dims=input_dims,
                            fc1_dims=64, fc2_dims=64)
        
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype= np.bool)
    
    def store_transition(self, state, action ,reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        
        self.mem_counter += 1
    
    def choose_action (self, observation):
        if np.random.random() > self.epsilon:
            state = torch.FloatTensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def train(self):
        if self.mem_counter < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem,self.batch_size, replace=False)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        action_batch = self.action_memory[batch]
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * torch.max(q_next,dim=1)[0]
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        