#!/usr/bin/env python3
# Tim Marvel
import math
import os
import time
import random
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
# custom
import src.tools
from src import reward_manager, tools, neural_net_lib, custom_data_loader_text


class CustomDatasetFromCSV(Dataset):
    def __init__(self, path_to_file):
        self.grid_values = []
        self.rewards = []
        with open(path_to_file) as file_obj:
            csv_obj = csv.reader(file_obj)
            for line in csv_obj:
                self.rewards.append(np.float32(line[9]))
                # TODO add rotation
                self.grid_values.append([line[0:3], line[3:6],line[6:9]])
        self.len = len(self.rewards)
        self.grid_values = torch.Tensor(np.array(np.float32(self.grid_values))).to(device)
        self.rewards = torch.Tensor(np.array(np.float32(self.rewards))).to(device)

    def __getitem__(self, item):
        return self.grid_values[item], self.rewards[item]

    def __len__(self):
        return self.len

# TODO: remove
class CustomBinaryDatasetFromCSV(Dataset):
    def __init__(self, path_to_file):
        self.grid_values = []
        self.rewards = []
        with open(path_to_file) as file_obj:
            csv_obj = csv.reader(file_obj)
            for line in csv_obj:
                rwrd = 1
                if(float(line[9])==float('0.0')):
                    rwrd = 0
                self.rewards.append(rwrd)
                self.grid_values.append([line[0:3], line[3:6], line[6:9]])
        self.len = len(self.rewards)
        self.grid_values = torch.Tensor(np.array(np.float32(self.grid_values))).to(device)
        self.rewards = torch.Tensor(np.array(np.float32(self.rewards))).to(device)

    def __getitem__(self, item):
        return self.grid_values[item], self.rewards[item]

    def __len__(self):
        return self.len


def plot_results(resluts_array,test_array, name_of_plot):
    plt.clf()
    plt.ylim(0, 15)
    plt.plot(np.array(resluts_array))
    plt.plot(np.array(test_array))
    plt.savefig("results/"+name_of_plot)

#80k epochs
def train_raw_three(epoch=5000, batch_size=32*4096, plot_result=True,
                            backup_name = "raw_net_three_probofchg_ff",
                            learning_rate = 4.5*0.0002,
                            use_pretrained = False,
                            pretrained_name = "none",
                            training_loss_graph = "raw_three_net_larger.png"):

    #prob_of_change_net = neural_net_lib.ThreeByThreeProbofchng1ConvLayer().to(device)
    prob_of_change_net = neural_net_lib.ThreeByThreeProbofchng1ConvLayerLarger().to(device)

    train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}

    dataset = CustomDatasetFromCSV('collected_data/unique_rotated_pts.csv')
    # split 80-20
    print(len(dataset))
    train_len = int (len(dataset)*.9);
    test_len = int (len(dataset)-train_len)
    train_data, test_data = random_split(dataset, [train_len, test_len])

    #dataset = CustomBinaryDatasetFromCSV('collected_data/unique_pts.csv')
    train_loader_three = DataLoader(train_data, **train_params)
    test_loader_three = DataLoader(test_data, **train_params)
    optimizer_three = optim.Adam(prob_of_change_net.parameters(), lr=learning_rate)
    # TODO investigate which loss is better
    # l1 loss = SmoothL1Loss is pytorch name for huber loss function
    l1_loss = nn.SmoothL1Loss()
    l2_loss = nn.MSELoss()
    # cross entropy only works with integers, check how this correlates with sigmoid at the end of the net
    # cross_ent_loss = nn.CrossEntropyLoss().to(device)
    '''
    if use_pretrained:
        backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(),
                                          ("../saved_nets/" + pretrained_name)))
        cluster_net_three.load_state_dict(torch.load(backup_net_name))
        cluster_net_three.to(device)
    '''
    train_losses = []
    eval_losses = []
    prob_of_change_net.train()
    start_time = time.time()
    for e in range (epoch):
        if e%50 == 0 and e>0:
            end_time = time.time()
            print("epoch: "+str(e))
            print(end_time-start_time)
            print("train losses last ")
            print(train_losses[-1])
            start_time = end_time
            if plot_result:
                plot_results(train_losses, eval_losses,"epoch_"+str(e)+"_"+training_loss_graph)
        train_loss_e = 0
        for i, data in enumerate(train_loader_three):
            inputs, rewards = data
            inputs = inputs.unsqueeze(1)
            rewards = rewards.to(torch.float)
            result = prob_of_change_net.forward(inputs)
            rewards = rewards.unsqueeze(1)
            train_loss = l1_loss(result, rewards)
            optimizer_three.zero_grad()
            train_loss.backward()
            optimizer_three.step()
            train_loss_e += train_loss.detach().cpu()
        train_losses.append(train_loss_e)
        eval_loss = 0
        for i, data in enumerate(test_loader_three):
            inputs, rewards = data
            inputs = inputs.unsqueeze(1)
            rewards = rewards.to(torch.float)
            result = prob_of_change_net.forward(inputs)
            rewards = rewards.unsqueeze(1)
            train_loss = l1_loss(result, rewards)
            eval_loss += train_loss.detach().cpu()
        eval_losses.append(eval_loss)
        # run evaluation

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(),("../saved_nets/"+backup_name)))
    torch.save(prob_of_change_net.state_dict(), backup_net_name)
    # training_plots
    if plot_result:
        print("plotting results")
        plt.clf()
        plt.xlim(0, 10)
        plt.plot(np.array(train_losses))
        plt.plot(np.array(eval_losses))
        plt.savefig(training_loss_graph)

device = tools.get_device()
train_raw_three()

