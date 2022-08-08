#!/usr/bin/env python3
# Tim Marvel
import math
import os
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


def train_cluster_net(epoch = 1000, batch_size=8192, plot_result=False, backup_name="backup_net"):
    cluster_net_three = neural_net_lib.ThreeByThreeCluster().to(device)
    params_cluster_three = {'batch_size': batch_size, f'shuffle': True, 'num_workers': 0}
    cluster_set_three = custom_data_loader_text.CustomDatasetFromTextFiles3(is_small=False, is_clean=True,
                                                                            with_var=True, cluster_num=-1)
    cluster_loader_three = DataLoader(cluster_set_three, **params_cluster_three)
    optimizer_cluster_three = optim.Adam(cluster_net_three.parameters(), lr=0.001)
    l1_loss = nn.SmoothL1Loss()
    train_losses = []
    for e in range (epoch):
        if e%100 == 0:
            print("epoch: "+str(e))
        for i, data in enumerate(cluster_loader_three):
            inputs, rewards = data
            # make sure it is the same length as batch size
            input_len = len(inputs)
            inputs = inputs.reshape([input_len, 3, 3]).to(device)
            rewards = rewards.reshape([input_len, 1]).to(device)
            result = cluster_net_three.forward(inputs)
            train_loss = l1_loss(result, rewards)
            optimizer_cluster_three.zero_grad()
            train_loss.backward()
            optimizer_cluster_three.step()
            train_losses.append(train_loss.detach().cpu())

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(),("../saved_nets/"+backup_name)))
    torch.save(cluster_net_three.state_dict(), backup_net_name)
    if plot_result:
        plt.plot(np.array(train_losses))
        plt.show()

def train_three_by_three_raw_net(epoch = 1000, batch_size=8192, plot_result=False, backup_name="backup_net"):
    neural_net_three = neural_net_lib.ThreeByThreeSig().to(device)
    params_three = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    train_dataset_three = custom_data_loader_text.CustomDatasetFromTextFiles3(is_small=False, is_clean=False,
                                                                            with_var=False, cluster_num=-2)
    train_loader_three = DataLoader(train_dataset_three, **params_three)
    l1_loss = nn.SmoothL1Loss()

    train_losses = []
    for e in range (epoch):
        if e%100 == 0:
            print("epoch: "+str(e))

device = tools.get_device()

