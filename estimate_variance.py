#!/usr/bin/env python3
# Tim Marvel
import math
import os
import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

import src.tools
from src import neural_net_lib, reward_manager, tools, custom_data_loader_text

def add_variance_and_cluster_three(thresh=0.3):
    neural_net = neural_net_lib.ThreeByThreeSig().to(device)
    net_name = os.path.abspath(os.path.join(tools.get_working_dir(), '../saved_nets/raw_net_three'))
    neural_net.load_state_dict(torch.load(net_name))

    # we want them 1 by 1 since we are writing in a datafile
    params_three = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}

    #true because is small dataset
    custom_set_three = custom_data_loader_text.CustomDatasetFromTextFiles3()
    train_loader_three = DataLoader(custom_set_three, **params_three)

    _rewards3_text_file_with_var = open(text_file_with_var3, 'w')

    for i, data in enumerate(train_loader_three):
        inputs, rewards = data
        # call reward shaper to perform comparison
        rewards = reward_manager.reward_shaper_three(rewards,inputs)
        # make sure it is the same length as batch size
        input_len = len(inputs)
        inputs_res = inputs.reshape([input_len, 3, 3])
        rewards = rewards.reshape([input_len, 1])
        result = neural_net.forward(inputs_res)
        cluster = 0

        if abs(result - rewards) > thresh:
            cluster = 1
            if result < -0.15 or rewards < -0.15:
                cluster = 2


        inputs_list = inputs.flatten().tolist()
        list = ','.join(str(v) for v in inputs_list)
        _rewards3_text_file_with_var.write(list+","+str(rewards.item())+","+str(result.item())+","+str(cluster)+"\n")


    _rewards3_text_file_with_var.close()


def add_variance_and_cluster_five_conv(backup_name="raw_net_five_conv", plot_result=False,  thresh=0.3):
    neural_net = neural_net_lib.FiveByFiveConv().to(device)
    net_name = os.path.abspath(os.path.join(tools.get_working_dir(), '../saved_nets/'+backup_name))
    neural_net.load_state_dict(torch.load(net_name))
    neural_net.eval()
    # we want them 1 by 1 since we are writting in a datafile
    params_three = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}

    custom_set_five = custom_data_loader_text.CustomDatasetFromTextFiles5()
    train_loader_five = DataLoader(custom_set_five, **params_three)

    _rewards5_text_file_with_var = open(text_file_with_var5, 'w')

    results_plot = []
    rewards_plot = []

    for i, data in enumerate(train_loader_five):
        inputs, rewards = data
        '''
        for i in range(len(rewards)):
            if rewards[i] == -10:
                rewards[i] = -64
        '''
        # make sure it is the same length as batch size
        input_len = len(inputs)
        inputs_res = inputs.reshape([input_len, 5, 5]).to(device)
        inputs_res = inputs_res.unsqueeze(1)
        # unsqueeze for convolutional neural net
        rewards = rewards.reshape([input_len, 1]).to(device)
        rewards_plot.append(rewards.item())
        result = neural_net.forward(inputs_res)
        results_plot.append(result.item())
        cluster = 0

        if abs(result - rewards) > thresh:
            cluster = 1
            if result < -0.15 or rewards < -0.15:
                cluster = 2


        inputs_list = inputs.flatten().tolist()
        list = ','.join(str(v) for v in inputs_list)
        _rewards5_text_file_with_var.write(list+","+str(rewards.item())+","+str(result.item())+","+str(cluster)+"\n")

    if plot_resul:
        plt.plot(results_plot)
        plt.plot(rewards_plot)
        plt.show()

    _rewards5_text_file_with_var.close()


text_file_with_var3 = tools.get_text_file_names_var()[0]
text_file_with_var5 = tools.get_text_file_names_var()[1]
#device = tools.get_device()
# everything is happening on the cpu since we are going 1 by 1
device = "cpu"
#add_variance_and_cluster_five()
