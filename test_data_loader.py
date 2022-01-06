import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import src.tools
from src import custom_data_loader, tools, neural_net_lib

import torch.optim as optim
import torch.nn as nn

# how to use batch size?
'''
# this works
number = 4.0
array_n = [4.0, 4.0, 4.0, 4.0]
ttensor = torch.Tensor(array_n)
print("this is number "+str(number))
print("this is array_n "+str(array_n))
print(ttensor)
'''
'''
pos_location, neg_location = tools.get_save_location_three()
src.tools.get_save_location_three()
'''

params_three = {'batch_size': 32,
                'shuffle': True,
                'num_workers': 0}

custom_set_three = custom_data_loader.CustomDataset(src.tools.get_save_location_three()[0],
                                                    src.tools.get_save_location_three()[1])
train_loader_three = DataLoader(custom_set_three, **params_three)

params_five = {'batch_size': 32,
               'shuffle': True,
               'num_workers': 0}

custom_set_five = custom_data_loader.CustomDataset(src.tools.get_save_location_five()[0],
                                                   src.tools.get_save_location_five()[1])
train_loader_five = DataLoader(custom_set_five, **params_three)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

neural_net_three = neural_net_lib.ThreeByThreeSig().to(device)
optimizer_three = optim.Adam(neural_net_three.parameters(), lr=0.0005)

neural_net_five = neural_net_lib.FiveByFiveSig().to(device)
optimizer_five = optim.Adam(neural_net_five.parameters(), lr=0.0007)

l1_loss = nn.SmoothL1Loss()

train_losses_three = []
train_losses_five = []

for e in range(1):
    for i, data in enumerate(train_loader_three):
        inputs, rewards = data
        # make sure it is the same length as batch size
        input_len = len(inputs)
        inputs = inputs.reshape([input_len * 4, 3, 3])
        rewards = rewards.reshape([input_len * 4, 1])
        result = neural_net_three.forward(inputs)
        train_loss = l1_loss(result, rewards)
        optimizer_three.zero_grad()
        train_loss.backward()
        optimizer_three.step()
        train_losses_three.append(train_loss)

for e in range(1):
    for i, data in enumerate(train_loader_five):
        inputs, rewards = data
        # make sure it is the same length as batch size
        input_len = len(inputs)
        inputs = inputs.reshape([input_len * 4, 5, 5])
        rewards = rewards.reshape([input_len * 4, 1])
        result = neural_net_five.forward(inputs)
        train_loss = l1_loss(result, rewards)
        optimizer_five.zero_grad()
        train_loss.backward()
        optimizer_five.step()
        train_losses_five.append(train_loss)

train_losses_three = np.array(train_losses_three)
plt.plot(train_losses_three)
plt.plot(np.array(train_losses_five))
#plt.legend()

plt.show()
