#!/usr/bin/env python3
# Tim Marvel
import math
import os
import time
import random
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
# ml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from sklearn.model_selection import KFold
# custom
import src.tools
from src import reward_manager, tools, model_zoo, custom_data_loader_text


# Silence external libs
logging.basicConfig(level=logging.CRITICAL, filename='logs/training_corss_validation_7by7.log', encoding='utf-8')
logger = logging.getLogger('train_cross_validation_7by7')
# enable logs for current lib
logger.setLevel(level=logging.INFO)


class CustomDatasetFromCSV(Dataset):
    def __init__(self, path_to_file):
        self.grid_values = []
        self.rewards = []
        with open(path_to_file) as file_obj:
            csv_obj = csv.reader(file_obj)
            for line in csv_obj:
                self.rewards.append(np.float32(line[49]))
                self.grid_values.append([line[0:7], line[7:14], line[14:21], line[21:28], line[28:35], line[35:42], line[42:49]])
        self.len = len(self.rewards)
        self.grid_values = torch.Tensor(np.array(np.float32(self.grid_values))).to(device)
        self.rewards = torch.Tensor(np.array(np.float32(self.rewards))).to(device)

    def __getitem__(self, item):
        return self.grid_values[item], self.rewards[item]

    def __len__(self):
        return self.len


def plot_train_loss_curves(resluts_array, test_array, name_of_plot):
    plt.clf()
    plt.plot(np.array(resluts_array), label='train error')
    plt.plot(np.array(test_array), label='test error')
    plt.legend()
    plt.savefig("training_plots/"+name_of_plot)


def train_net(epoch = 20,
              batch_size = 32,
              neural_net_size = 32,
              dropout = 0.0,
              learning_rate = 0.001,
              plot_result = True,
              use_pretrained = False,
              pretrained_name = "none"):
    k_folds = 5
    kf_split = 1.0/k_folds
    kf = KFold(n_splits=k_folds, shuffle=True)
    backup_name = 'seven_conv_{}_drop_{}_bs_{}_m25_nd_l1'.format(neural_net_size, int(100.0*dropout), batch_size)
    training_loss_graph = backup_name+".png"
    print(backup_name)
    logger.info('Training net name: {}'.format(backup_name))
    #net = model_zoo.SevenBySeven1ConvLayerXLeakyReLUSigmoidEnd(neural_net_size, dropout).to(device)
    net = model_zoo.SevenBySeven2ConvLayerXLeakyReLUSigmoidEnd(neural_net_size, dropout).to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # keep both loss functions available for experimentation
    l1_loss = nn.SmoothL1Loss()
    l2_loss = nn.MSELoss()

    # these datasets contain different types of normalization / standardization
    dataset = CustomDatasetFromCSV('collected_data/unique_normalized_7_rewards_m25.csv')

    train_losses = []
    eval_losses = []
    best_loss = 100
    for j in range(2):
        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
            logger.info("fold f: {}".format(fold))
            train_loader_five = DataLoader(dataset, batch_size = batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx), shuffle = True)
            test_loader_five = DataLoader(dataset, batch_size = batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_idx), shuffle = True)
            train_len = len(train_loader_five.dataset)

            for e in range(epoch):
                train_loss_e = 0
                net.train()
                for i, data in enumerate(train_loader_five):
                    inputs, rewards = data
                    inputs = inputs.unsqueeze(1)
                    rewards = rewards.to(torch.float)
                    result = net.forward(inputs)
                    rewards = rewards.unsqueeze(1)
                    train_loss = l1_loss(result, rewards)
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                    train_loss_e += train_loss.detach().cpu()
                train_losses.append(train_loss_e/(kf_split*(k_folds-1)*train_len))

                net.eval()
                eval_loss_e = 0
                for i, data in enumerate(test_loader_five):
                    inputs, rewards = data
                    inputs = inputs.unsqueeze(1)
                    rewards = rewards.to(torch.float)
                    result = net.forward(inputs)
                    rewards = rewards.unsqueeze(1)
                    eval_loss = l1_loss(result, rewards)
                    eval_loss_e += train_loss.detach().cpu()
                
                if best_loss>eval_loss and epoch>0:
                    best_loss =  eval_loss
                    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + backup_name+"_best")))
                    torch.save(net.state_dict(), backup_net_name)
                
                eval_losses.append(eval_loss_e/(kf_split*train_len))

                if plot_result and e%5==0 and e>0:
                    plot_train_loss_curves(train_losses, eval_losses, "iter_"+str(j)+"kfold_"+str(fold)+ "_epoch_" + str(e) +"_" + training_loss_graph)
                    plot_train_loss_curves(train_losses, train_losses, "iter_"+str(j)+"kfold_"+str(fold)+ "_epoch_" + str(e) +"_onlytrain_" + training_loss_graph)
    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + backup_name)))
    torch.save(net.state_dict(), backup_net_name)



def train_net_simple(epoch = 20,
              batch_size = 32,
              neural_net_size = 32,
              dropout = 0.0,
              learning_rate = 0.001,
              plot_result = True,
              use_pretrained = False,
              pretrained_name = "none"):
    backup_name = 'seven_conv_{}_drop_{}_bs_{}_m25_nd_l1'.format(neural_net_size, int(100.0*dropout), batch_size)
    training_loss_graph = backup_name+".png"
    print(backup_name)
    logger.info('Training net name: {} without kfolds'.format(backup_name))
    net = model_zoo.SevenBySeven2ConvLayerXLeakyReLUSigmoidEnd(neural_net_size, dropout).to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # keep both loss functions available for experimentation
    l1_loss = nn.SmoothL1Loss()
    l2_loss = nn.MSELoss()

    # these datasets contain different types of normalization / standardization
    dataset = CustomDatasetFromCSV('collected_data/unique_normalized_7_rewards_m25.csv')

    train_losses = []
    eval_losses = []
    best_loss = 100
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    train_loader_seven = DataLoader(dataset, **params)
    train_len = len(train_loader_seven.dataset)
    for e in range(epoch):
        train_loss_e = 0
        net.train()
        for i, data in enumerate(train_loader_seven):
            inputs, rewards = data
            inputs = inputs.unsqueeze(1)
            rewards = rewards.to(torch.float)
            result = net.forward(inputs)
            rewards = rewards.unsqueeze(1)
            train_loss = l1_loss(result, rewards)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_loss_e += train_loss.detach().cpu()
        train_losses.append(train_loss_e/train_len)
        
        if plot_result and e%5==0 and e>0:
            plot_train_loss_curves(train_losses, train_losses, "_epoch_" + str(e) +"_" + training_loss_graph)
                
    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + backup_name)))
    torch.save(net.state_dict(), backup_net_name)


device = tools.get_device()
logger.info('training with device {}'.format(device))


#train_net(epoch = 6, learning_rate=0.0008, neural_net_size = 32, batch_size = 32)
#train_net(epoch = 6, learning_rate=0.0008, neural_net_size = 32, batch_size = 128)
#train_net_simple(epoch = 51, learning_rate=0.001, neural_net_size = 32, dropout=0.00, batch_size = 128)
train_net_simple(epoch = 41, learning_rate=0.001, neural_net_size = 32, dropout=0.00, batch_size = 128)
#train_net_simple(epoch = 31, learning_rate=0.001, neural_net_size = 32, dropout=0.00, batch_size = 128)
#train_net_simple(epoch = 51, learning_rate=0.001, neural_net_size = 16, dropout=0.00, batch_size = 16384)

#train_net_simple(epoch = 141, learning_rate=0.0005, neural_net_size = 16, dropout=0.01, batch_size = 32)

#train_net(epoch = 21, learning_rate=0.0008, neural_net_size = 8, batch_size = 32)
#train_net(epoch = 61, learning_rate=0.0008, neural_net_size = 32, batch_size = 64)
#train_net(epoch = 61, learning_rate=0.0008, neural_net_size = 32, batch_size = 1024*32)
#train_net(epoch = 41, learning_rate=0.0008, neural_net_size = 8, batch_size = 2048*32)
#train_net(epoch = 41, learning_rate=0.0008, neural_net_size = 32, batch_size = 32)
#train_net(epoch = 41, learning_rate=0.008, neural_net_size = 32, batch_size = 32)
#train_net(epoch = 121, learning_rate=0.0008, neural_net_size = 256, batch_size = 16384)
#train_net(epoch = 61, learning_rate=0.00008, neural_net_size = 64, batch_size = 131072)
#train_net(epoch = 121, learning_rate=0.0008, neural_net_size = 512, batch_size = 8*16384)