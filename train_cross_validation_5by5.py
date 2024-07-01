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
logging.basicConfig(level=logging.CRITICAL, filename='logs/training_corss_validation_5by5.log', encoding='utf-8')
logger = logging.getLogger('train_cross_validation_5by5')
# enable logs for current lib
logger.setLevel(level=logging.INFO)


class CustomDatasetFromCSV(Dataset):
    def __init__(self, path_to_file):
        self.grid_values = []
        self.rewards = []
        with open(path_to_file) as file_obj:
            csv_obj = csv.reader(file_obj)
            for line in csv_obj:
                self.rewards.append(np.float32(line[25]))
                self.grid_values.append([line[0:5], line[5:10], line[10:15], line[15:20], line[20:25]])
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
    backup_name = 'five_conv_{}_drop_{}_bs_{}_m25_nd_l1'.format(neural_net_size, int(100.0*dropout), batch_size)
    training_loss_graph = backup_name+".png"
    print(backup_name)
    logger.info('Training net name: {}'.format(backup_name))
    net = model_zoo.FiveByFive1ConvLayerX(neural_net_size, dropout).to(device)
    optimizer_three = optim.Adam(net.parameters(), lr=learning_rate)

    # keep both loss functions available for experimentation
    l1_loss = nn.SmoothL1Loss()
    l2_loss = nn.MSELoss()

    # these datasets contain different types of normalization / standardization
    dataset = CustomDatasetFromCSV('collected_data/unique_normalized_5_rewards_m25.csv')

    train_losses = []
    eval_losses = []
    for j in range(2):
        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
            logger.info("fold f: {}".format(fold))
            train_loader_five = DataLoader(dataset, batch_size = batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
            test_loader_five = DataLoader(dataset, batch_size = batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_idx))
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
                    optimizer_three.zero_grad()
                    train_loss.backward()
                    optimizer_three.step()
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
                eval_losses.append(eval_loss_e/(kf_split*train_len))

                if plot_result and e%10==0 and e>0:
                    plot_train_loss_curves(train_losses, eval_losses, "iter_"+str(j)+"kfold_"+str(fold)+ "_epoch_" + str(e) +"_" + training_loss_graph)

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + backup_name)))
    torch.save(net.state_dict(), backup_net_name)

device = tools.get_device()
logger.info('training with device {}'.format(device))

train_net(epoch = 61, learning_rate=0.0008, neural_net_size = 256, batch_size = 16)
