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
pos_location, neg_location = tools.get_save_location_three()
src.tools.get_save_location_three()
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 0}
custom_set = custom_data_loader.CustomDataset(src.tools.get_save_location_three()[0],
                                              src.tools.get_save_location_three()[1])
train_loader = DataLoader(custom_set, **params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
neural_net = neural_net_lib.ThreeByThreeSig().to(device)
optimizer = optim.Adam(neural_net.parameters(), lr=0.0001)
l1_loss = nn.SmoothL1Loss()


train_losses = []
print("started ")
for e in range(2):
    for i, data in enumerate(train_loader):
        inputs, rewards = data
        # make sure it is the same length as batch size
        input_len = len(inputs)
        inputs = inputs.reshape([input_len*4, 3, 3])
        rewards = rewards.reshape([input_len*4, 1])
        result = neural_net.forward(inputs)
        train_loss = l1_loss(result, rewards)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss)

train_losses = np.array(train_losses)
plt.plot(train_losses)
plt.show()
#exit()


'''
print("test 3")
exit()
for i in range(0, 2):
    a = custom_set[i]
    print(a)

print("test 3")
# pos_location, neg_location = tools.get_save_test_location_three()
custom_set = custom_data_loader.CustomDataset(tools.get_save_test_location_three()[0],
                                              tools.get_save_test_location_three()[1])
for i in range(0, 2):
    a = custom_set[i]
    print(a)
print("test 4")
custom_set = custom_data_loader.CustomDataset(tools.get_save_test_location_five()[0],
                                              tools.get_save_test_location_five()[1])
for i in range(0, 2):
    a = custom_set[i]
    print(a)
print("test 5")
custom_set = custom_data_loader.CustomDataset(tools.get_save_location_five()[0],
                                              tools.get_save_location_five()[1])
for i in range(0, 2):
    a = custom_set[i]
    print(a)
print("test 6")
'''