#!/usr/bin/env python3
# Tim Marvel

import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import src.tools
from src import custom_data_loader_text, tools, model_zoo, reward_manager

class trainThreeByThree():
    train_losses = []

    def __init__(self, learning_rate, batch_size):
        self.train_param = {'batch_size': batch_size,
                    'shuffle': True, 'num_workers': 0}
        self.dataset = custom_data_loader_text.CustomDatasetFromTextFiles3()
        self.data_loader = DataLoader(self.dataset, **self.train_param)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neural_net = model_zoo.ThreeByThreeSig().to(self.device)
        self.optimizer = optim.Adam(self.neural_net.parameters(), lr=learning_rate)
        self.train_losses = []
        self.lossfunction = nn.SmoothL1Loss()


    def train(self, iterations, clearData = False):
        print(iterations)
        if clearData:
            self.train_losses = []

        for e in range(iterations):
            for i, data in enumerate(self.data_loader):
                inputs, rewards = data
                #print(rewards)
                #print(inputs)
                # reward shaping
                input_len = len(inputs)
                inputs = inputs.reshape([input_len, 3, 3])
                rewards = reward_manager.reward_shaper_three(rewards,inputs)
                inputs = inputs.to(self.device)
                rewards = rewards.reshape([input_len, 1]).to(self.device)
                result = self.neural_net.forward(inputs)
                train_loss = self.lossfunction(result, rewards)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                self.train_losses.append(train_loss)

    def saveNetName(self, name):
        path=os.path.abspath(os.path.join(tools.get_working_dir(), "../saved_nets/neural_net_three_"+str(name)))
        torch.save(self.neural_net.state_dict(), path)


class trainFiveByFive():
    train_losses = []

    def __init__(self, learning_rate, batch_size):
        self.train_param = {'batch_size': batch_size,
                    'shuffle': True, 'num_workers': 0}
        self.dataset = custom_data_loader_text.CustomDatasetFromTextFiles5()
        self.data_loader = DataLoader(self.dataset, **self.train_param)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neural_net = model_zoo.FiveByFiveSig().to(self.device)
        self.optimizer = optim.Adam(self.neural_net.parameters(), lr=learning_rate)
        self.train_losses = []
        self.lossfunction = nn.SmoothL1Loss()


    def train(self, iterations, clearData = False):
        print(iterations)
        if clearData:
            self.train_losses = []

        for e in range(iterations):
            for i, data in enumerate(self.data_loader):
                inputs, rewards = data
                # reward shaping
                # rewards =
                input_len = len(inputs)
                inputs = inputs.reshape([input_len, 5, 5]).to(self.device)
                rewards = rewards.reshape([input_len, 1]).to(self.device)
                result = self.neural_net.forward(inputs)
                train_loss = self.lossfunction(result, rewards)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                self.train_losses.append(train_loss)

    def saveNetName(self, name):
        path=os.path.abspath(os.path.join(tools.get_working_dir(), "../saved_nets/neural_net_five_"+str(name)))
        torch.save(self.neural_net.state_dict(), path)




'''
params_three = {'batch_size': 32,
                'shuffle': True,
                'num_workers': 0}


custom_set_three = custom_data_loader_text.CustomDatasetFromTextFiles3(True)
train_loader_three = DataLoader(custom_set_three, **params_three)

params_five = {'batch_size': 32,
               'shuffle': True,
               'num_workers': 0}

custom_set_five = custom_data_loader_text.CustomDatasetFromTextFiles5(True)
train_loader_five = DataLoader(custom_set_five, **params_three)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

neural_net_three = model_zoo.ThreeByThreeSig().to(device)
optimizer_three = optim.Adam(neural_net_three.parameters(), lr=0.005)

neural_net_five = model_zoo.FiveByFiveSig().to(device)
optimizer_five = optim.Adam(neural_net_five.parameters(), lr=0.005)

l1_loss = nn.SmoothL1Loss()

train_losses_three = []
train_losses_five = []

for e in range(2000):
    for i, data in enumerate(train_loader_three):
        inputs, rewards = data
        # make sure it is the same length as batch size
        input_len = len(inputs)
        inputs = inputs.reshape([input_len, 3, 3]).to(device)
        rewards = rewards.reshape([input_len, 1]).to(device)
        result = neural_net_three.forward(inputs)
        train_loss = l1_loss(result, rewards)
        optimizer_three.zero_grad()
        train_loss.backward()
        optimizer_three.step()
        train_losses_three.append(train_loss)

for e in range(2000):
    for i, data in enumerate(train_loader_five):
        inputs, rewards = data
        # make sure it is the same length as batch size
        input_len = len(inputs)
        inputs = inputs.reshape([input_len, 5, 5]).to(device)
        rewards = rewards.reshape([input_len, 1]).to(device)
        result = neural_net_five.forward(inputs)
        train_loss = l1_loss(result, rewards)
        optimizer_five.zero_grad()
        train_loss.backward()
        optimizer_five.step()
        train_losses_five.append(train_loss)


backup_net_name_five = os.path.abspath(
        os.path.join(tools.get_working_dir(), "../saved_nets/neural_net_five_test_five"))

backup_net_name_three = os.path.abspath(
        os.path.join(tools.get_working_dir(), "../saved_nets/neural_net_five_test_five"))

torch.save(neural_net_five.state_dict(), backup_net_name_five)
torch.save(neural_net_three.state_dict(), backup_net_name_three)

#train_losses_three = np.array(train_losses_three)
plt.plot(np.array(train_losses_three))
plt.plot(np.array(train_losses_five))

plt.show()
'''
