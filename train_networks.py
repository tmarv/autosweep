#!/usr/bin/env python3
# Tim Marvel
import math
import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# ml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# custom
import src.tools
from src import reward_manager, tools, neural_net_lib, custom_data_loader_text


def train_cluster_net_three(epoch=1000, batch_size=8192, plot_result=False, backup_name="backup_net", learning_rate=0.001):
    print("this is device "+str(device))
    cluster_net_three = neural_net_lib.ThreeByThreeCluster().to(device)
    params_cluster_three = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    cluster_set_three = custom_data_loader_text.CustomDatasetFromTextFiles3(is_small=False, is_clean=True,
                                                                            with_var=True, cluster_num=-1, device=device)
    cluster_loader_three = DataLoader(cluster_set_three, **params_cluster_three)
    optimizer_cluster_three = optim.Adam(cluster_net_three.parameters(), lr=learning_rate)
    l1_loss = nn.SmoothL1Loss().to(device)
    train_losses = []
    cluster_net_three.train()
    start_time = time.time()
    for e in range (epoch):
        if e%100 == 0:
            end_time = time.time()
            print("epoch: "+str(e))
            print(end_time-start_time)
            start_time=end_time
        for i, data in enumerate(cluster_loader_three):
            inputs, clusters = data
            #clusters = torch.from_numpy(clusters)
            #print("len afore: " + str(len(clusters)))
            # make sure it is the same length as batch size
            #input_len = len(inputs)
            #inputs = inputs.reshape([input_len, 3, 3]).to(device)
            #inputs = inputs.reshape([input_len, 3, 3])
            #clusters = clusters.reshape([input_len, 3]).to(device)
            #clusters = clusters.reshape([input_len, 3])
            result = cluster_net_three.forward(inputs)
            train_loss = l1_loss(result, clusters)
            optimizer_cluster_three.zero_grad()
            train_loss.backward()
            optimizer_cluster_three.step()
            train_losses.append(train_loss.detach().cpu())

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(),("../saved_nets/"+backup_name)))
    torch.save(cluster_net_three.state_dict(), backup_net_name)
    plt.plot(np.array(train_losses))
    plt.savefig(backup_name+".png")
    if plot_result:
        plt.show()


def train_cluster_net_five_conv(epoch=1000, batch_size=8192, plot_result=False, backup_name="backup_net_cluster_five", learning_rate=0.001):
    print("this is device " + str(device))
    cluster_net_five_conv = neural_net_lib.FiveByFiveConv().to(device)
    params_cluster_five_conv = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    cluster_set_five_conv = custom_data_loader_text.CustomDatasetFromTextFiles5(is_small=False, is_clean=True,
                                                                            with_var=True, cluster_num=-1,
                                                                            device=device)
    cluster_loader_five_conv = DataLoader(cluster_set_five_conv, **params_cluster_five_conv)
    optimizer_cluster_five_conv = optim.Adam(cluster_net_five_conv.parameters(), lr=learning_rate)
    l1_loss = nn.SmoothL1Loss().to(device)
    train_losses = []
    cluster_net_five_conv.train()
    start_time = time.time()
    for e in range(epoch):
        if e%100 == 0:
            end_time = time.time()
            print("epoch: "+str(e))
            print(end_time-start_time)
            start_time = end_time
        for i, data in enumerate(cluster_loader_five_conv):
            inputs, clusters = data
            # unsqueeze the data?
            result = cluster_net_five_conv.forward(inputs)
            train_loss = l1_loss(result, clusters)
            optimizer_cluster_five_conv.zero_grad()
            train_loss.backward()
            optimizer_cluster_five_conv.step()
            train_losses.append(train_loss.detach().cpu())

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + backup_name)))
    torch.save(cluster_net_five_conv.state_dict(), backup_net_name)
    plt.plot(np.array(train_losses))
    plt.savefig(backup_name + ".png")
    if plot_result:
        plt.show()
def train_three_by_three_raw_net(epoch = 1000, batch_size=8192, plot_result=False, backup_name="backup_net", learning_rate=0.001):
    neural_net_three = neural_net_lib.ThreeByThreeSig().to(device)
    params_three = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    train_dataset_three = custom_data_loader_text.CustomDatasetFromTextFiles3(is_small=False, is_clean=False,
                                                                            with_var=False, cluster_num=-2)
    train_loader_three = DataLoader(train_dataset_three, **params_three)
    optimizer_three = optim.Adam(neural_net_three.parameters(), lr=learning_rate)
    l1_loss = nn.SmoothL1Loss()
    train_losses = []

    for e in range (epoch):
        if e%100 == 0:
            print("epoch: "+str(e))
        for i, data in enumerate(train_loader_three):
            inputs, rewards = data
            # make sure it is the same length as batch size
            #input_len = len(inputs)
            #inputs = inputs.reshape([input_len, 3, 3]).to(device)
            #rewards = rewards.reshape([input_len, 1]).to(device)
            result = neural_net_three.forward(inputs)
            train_loss = l1_loss(result, rewards)
            optimizer_three.zero_grad()
            train_loss.backward()
            optimizer_three.step()
            train_losses.append(train_loss.detach().cpu())

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + backup_name)))
    torch.save(neural_net_three.state_dict(), backup_net_name)

    plt.plot(np.array(train_losses))
    plt.savefig(backup_name+".png")
    if plot_result:
        plt.show()


def train_three_by_three_for_one_cluster(cluster, epoch = 1000, batch_size = 2048, plot_result = False,
                                         backup_name = "backup_net", learning_rate = 0.001):
    neural_net_three = neural_net_lib.ThreeByThreeSig().to(device)
    params_three = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    train_dataset_three = custom_data_loader_text.CustomDatasetFromTextFiles3(is_small = False, is_clean = False,
                                                                              with_var = True, cluster_num = cluster,
                                                                              device = device)
    train_loader_three = DataLoader(train_dataset_three, **params_three)
    optimizer_three = optim.Adam(neural_net_three.parameters(), lr=learning_rate)
    l1_loss = nn.SmoothL1Loss()
    train_losses = []

    for e in range (epoch):
        if e%100 == 0:
            print("epoch: "+str(e))
        for i, data in enumerate(train_loader_three):
            inputs, rewards = data
            # make sure it is the same length as batch size
            # input_len = len(inputs)
            # inputs = inputs.reshape([input_len, 3, 3]).to(device)
            # rewards = rewards.reshape([input_len, 1]).to(device)
            result = neural_net_three.forward(inputs)
            train_loss = l1_loss(result, rewards)
            optimizer_three.zero_grad()
            train_loss.backward()
            optimizer_three.step()
            train_losses.append(train_loss.detach().cpu())

    backup_name+=("_cluster_"+str(cluster))
    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + backup_name)))
    torch.save(neural_net_three.state_dict(), backup_net_name)


    plt.plot(np.array(train_losses))
    plt.savefig(backup_name+".png")
    if plot_result:
        plt.show()


def train_five_by_five_raw_net(epoch = 1000, batch_size=8192, plot_result=False, backup_name="backup_net_five", learning_rate=0.001):
    neural_net_five = neural_net_lib.FiveByFiveSig().to(device)
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    train_dataset_five = custom_data_loader_text.CustomDatasetFromTextFiles5(is_small=False)
    train_loader_five = DataLoader(train_dataset_five, **params)
    optimizer_five = optim.Adam(neural_net_five.parameters(), lr=learning_rate)
    l1_loss = nn.SmoothL1Loss()
    train_losses = []
    start_time = time.time()

    for e in range (epoch):
        if e%100 == 0:
            end_time = time.time()
            print("epoch: " + str(e))
            print(end_time - start_time)
            start_time = end_time
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
            train_losses.append(train_loss.detach().cpu())

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + backup_name)))
    torch.save(neural_net_five.state_dict(), backup_net_name)

    plt.plot(np.array(train_losses))
    plt.savefig(backup_name+".png")
    if plot_result:
        plt.show()


def train_five_by_five_conv(epoch = 1000, batch_size=8192, plot_result=False, backup_name="backup_conv_net_five", learning_rate=0.001):
    print("training convolution net 5 by 5")
    neural_net_five_conv = neural_net_lib.FiveByFiveConv().to(device)
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    train_dataset_five = custom_data_loader_text.CustomDatasetFromTextFiles5(is_small=False)
    train_loader_five = DataLoader(train_dataset_five, **params)
    optimizer_five = optim.Adam(neural_net_five_conv.parameters(), lr=learning_rate)
    l1_loss = nn.SmoothL1Loss()
    train_losses = []
    start_time = time.time()
    for e in range(epoch):
        if e%100 == 0:
            end_time = time.time()
            print("epoch: " + str(e))
            print(end_time - start_time)
            start_time = end_time
        for i, data in enumerate(train_loader_five):
            inputs, rewards = data
            # make sure it is the same length as batch size
            input_len = len(inputs)
            inputs = inputs.reshape([input_len, 5, 5]).to(device)
            # add a dimension for the cnn
            inputs = inputs.unsqueeze(1)
            rewards = rewards.reshape([input_len, 1]).to(device)
            result = neural_net_five_conv.forward(inputs)
            train_loss = l1_loss(result, rewards)
            neural_net_five_conv.zero_grad()
            train_loss.backward()
            optimizer_five.step()
            train_losses.append(train_loss.detach().cpu())

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + backup_name)))
    torch.save(neural_net_five_conv.state_dict(), backup_net_name)

    plt.plot(np.array(train_losses))
    plt.savefig(backup_name+".png")
    if plot_result:
        plt.show()

#initialize everything
device = tools.get_device()
print(device)

