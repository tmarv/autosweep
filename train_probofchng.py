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
                self.grid_values.append([line[0:3], line[3:6],line[6:9]])
        self.len = len(self.rewards)
        self.grid_values = torch.Tensor(np.array(np.float32(self.grid_values))).to(device)
        self.rewards = torch.Tensor(np.array(np.float32(self.rewards))).to(device)

    def __getitem__(self, item):
        return self.grid_values[item], self.rewards[item]

    def __len__(self):
        return self.len

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


def train_probofchng_three(epoch=3000, batch_size=4096, plot_result=True,
                            backup_name = "probofchng_three_net_cross_ent",
                            learning_rate = 0.008,
                            use_pretrained = False,
                            pretrained_name = "none",
                            training_loss_graph = "probofchng_three_larger.png"):

    #prob_of_change_net = neural_net_lib.ThreeByThreeProbofchng1ConvLayer().to(device)
    prob_of_change_net = neural_net_lib.ThreeByThreeProbofchng1ConvLayerLarger().to(device)

    train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}

    #dataset = CustomDatasetFromCSV('collected_data/unique_pts.csv')
    dataset = CustomBinaryDatasetFromCSV('collected_data/unique_pts.csv')
    loader_three = DataLoader(dataset, **train_params)
    optimizer_three = optim.Adam(prob_of_change_net.parameters(), lr=learning_rate)
    # TODO investigate which loss is better
    #l1_loss = nn.SmoothL1Loss()
    #l2_loss = nn.MSELoss()
    # cross entropy only works with integers, check how this correlates with sigmoid at the end of the net
    cross_ent_loss = nn.CrossEntropyLoss().to(device)
    '''
    if use_pretrained:
        backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(),
                                          ("../saved_nets/" + pretrained_name)))
        cluster_net_three.load_state_dict(torch.load(backup_net_name))
        cluster_net_three.to(device)
    '''
    train_losses = []
    prob_of_change_net.train()
    start_time = time.time()
    for e in range (epoch):
        if e%100 == 0 and e>0:
            end_time = time.time()
            print("epoch: "+str(e))
            print(end_time-start_time)
            print("train losses last ")
            print(train_losses[-1])
            start_time = end_time
        for i, data in enumerate(loader_three):
            inputs, rewards = data
            inputs = inputs.unsqueeze(1)
            rewards = rewards.to(torch.float)
            result = prob_of_change_net.forward(inputs)
            rewards = rewards.unsqueeze(1)
            train_loss = l2_loss(result, rewards)
            optimizer_three.zero_grad()
            train_loss.backward()
            optimizer_three.step()
            train_losses.append(train_loss.detach().cpu())

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(),("../saved_nets/"+backup_name)))
    torch.save(prob_of_change_net.state_dict(), backup_net_name)

    if plot_result:
        plt.clf()
        plt.plot(np.array(train_losses))
        plt.savefig(training_loss_graph)


def plot_results(resluts_array, name_of_plot):
    plt.clf()
    plt.ylim(0, 10)
    plt.plot(np.array(resluts_array))
    plt.savefig("results/"+name_of_plot)


def train_raw_three(epoch=80000, batch_size=32*4096, plot_result=True,
                            backup_name = "raw_net_three_probofchg",
                            learning_rate = 4.5*0.0002,
                            use_pretrained = False,
                            pretrained_name = "none",
                            training_loss_graph = "raw_three_net_larger.png"):

    #prob_of_change_net = neural_net_lib.ThreeByThreeProbofchng1ConvLayer().to(device)
    prob_of_change_net = neural_net_lib.ThreeByThreeProbofchng1ConvLayerLarger().to(device)

    train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}

    dataset = CustomDatasetFromCSV('collected_data/unique_pts.csv')
    #dataset = CustomBinaryDatasetFromCSV('collected_data/unique_pts.csv')
    loader_three = DataLoader(dataset, **train_params)
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
    prob_of_change_net.train()
    start_time = time.time()
    for e in range (epoch):
        if e%500 == 0 and e>0:
            end_time = time.time()
            print("epoch: "+str(e))
            print(end_time-start_time)
            print("train losses last ")
            print(train_losses[-1])
            start_time = end_time
            if plot_result:
                plot_results(train_losses,"epoch_"+str(e)+"_"+training_loss_graph)
        for i, data in enumerate(loader_three):
            inputs, rewards = data
            inputs = inputs.unsqueeze(1)
            rewards = rewards.to(torch.float)
            result = prob_of_change_net.forward(inputs)
            rewards = rewards.unsqueeze(1)
            train_loss = l1_loss(result, rewards)
            optimizer_three.zero_grad()
            train_loss.backward()
            optimizer_three.step()
            train_losses.append(train_loss.detach().cpu())

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(),("../saved_nets/"+backup_name)))
    torch.save(prob_of_change_net.state_dict(), backup_net_name)

    if plot_result:
        plt.clf()
        plt.xlim(0, 10)
        plt.plot(np.array(train_losses))
        plt.savefig(training_loss_graph)

device = tools.get_device()
train_raw_three()
