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


def train_cluster_net_three(epoch=1000, batch_size=8192, plot_result=False,
                            backup_name="backup_net",
                            learning_rate=0.001,
                            use_pretrained = False,
                            pretrained_name="none",
                            training_loss_graph="graph.png"):
    cluster_net_three = neural_net_lib.ThreeByThreeCluster().to(device)
    params_cluster_three = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    cluster_set_three = custom_data_loader_text.CustomDatasetFromTextFiles3(is_small = False,
                                                                            is_clean = True,
                                                                            with_var = True,
                                                                            cluster_num = -1,
                                                                            device = device)
    cluster_loader_three = DataLoader(cluster_set_three, **params_cluster_three)
    optimizer_cluster_three = optim.Adam(cluster_net_three.parameters(), lr=learning_rate)
    cross_ent_loss = nn.CrossEntropyLoss().to(device)
    if use_pretrained:
        backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(),
                                          ("../saved_nets/" + pretrained_name)))
        cluster_net_three.load_state_dict(torch.load(backup_net_name))
        cluster_net_three.to(device)

    train_losses = []
    cluster_net_three.train()
    start_time = time.time()
    for e in range (epoch):
        if e%100 == 0:
            end_time = time.time()
            print("epoch: "+str(e))
            print(end_time-start_time)
            start_time = end_time
        for i, data in enumerate(cluster_loader_three):
            inputs, clusters = data
            result = cluster_net_three.forward(inputs)
            train_loss = cross_ent_loss(result, clusters)
            optimizer_cluster_three.zero_grad()
            train_loss.backward()
            optimizer_cluster_three.step()
            train_losses.append(train_loss.detach().cpu())

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(),("../saved_nets/"+backup_name)))
    torch.save(cluster_net_three.state_dict(), backup_net_name)

    if plot_result:
        plt.clf()
        plt.plot(np.array(train_losses))
        plt.savefig(training_loss_graph)


def train_cluster_net_five_conv(epoch = 1000,
                                batch_size = 8192,
                                plot_result = False,
                                backup_name = "backup_net_cluster_five",
                                learning_rate = 0.001,
                                use_pretrained = False,
                                pretrained_net_name = "backup_net_cluster_five",
                                training_loss_graph = "training_plots/train_cluster_five.png"):
    cluster_net_five_conv = neural_net_lib.FiveByFiveConvCluster().to(device)

    if use_pretrained:
        backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(),
                                          ("../saved_nets/" + pretrained_net_name)))
        cluster_net_five_conv.load_state_dict(torch.load(backup_net_name))
        cluster_net_five_conv.to(device)

    params_cluster_five_conv = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    cluster_set_five_conv = custom_data_loader_text.CustomDatasetFromTextFiles5(is_small=False,
                                                                                is_clean=True,
                                                                                with_var=True,
                                                                                cluster_num=-1,
                                                                                device=device)
    cluster_loader_five_conv = DataLoader(cluster_set_five_conv, **params_cluster_five_conv)
    optimizer_cluster_five_conv = optim.Adam(cluster_net_five_conv.parameters(), lr=learning_rate)
    # TODO make this a parameter
    cross_ent_loss = nn.CrossEntropyLoss().to(device)
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
            input_len = len(inputs)
            inputs = inputs.reshape(input_len, 5, 5)
            inputs = inputs.unsqueeze(1)
            result = cluster_net_five_conv.forward(inputs)
            train_loss = cross_ent_loss(result, clusters)
            optimizer_cluster_five_conv.zero_grad()
            train_loss.backward()
            optimizer_cluster_five_conv.step()
            train_losses.append(train_loss.detach().cpu())

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + backup_name)))
    torch.save(cluster_net_five_conv.state_dict(), backup_net_name)

    if plot_result:
        plt.clf()
        plt.plot(np.array(train_losses))
        plt.savefig(training_loss_graph)
def train_three_by_three_raw_net(epoch = 1000,
                                 batch_size=8192,
                                 plot_result=False,
                                 backup_name="backup_net",
                                 learning_rate=0.001,
                                 graph_name = "training_plots/train.png",
                                 use_pretrained_net = False,
                                 pretrained_net_name = "none"):
    neural_net_three = neural_net_lib.ThreeByThreeSig().to(device)
    params_three = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    train_dataset_three = custom_data_loader_text.CustomDatasetFromTextFiles3(is_small=False, is_clean=False,
                                                                            with_var=False, cluster_num=-2,
                                                                              device=device)
    train_loader_three = DataLoader(train_dataset_three, **params_three)
    optimizer_three = optim.Adam(neural_net_three.parameters(), lr=learning_rate)
    if use_pretrained_net:
        pretrained_net_path = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + pretrained_net_name)))
        neural_net_three.load_state_dict(torch.load(pretrained_net_path))
        neural_net_three.to(device)

    l1_loss = nn.SmoothL1Loss().to(device)
    train_losses = []

    for e in range (epoch):
        if e%100 == 0:
            print("epoch: "+str(e))
        for i, data in enumerate(train_loader_three):
            inputs, rewards = data
            result = neural_net_three.forward(inputs)
            train_loss = l1_loss(result, rewards)
            optimizer_three.zero_grad()
            train_loss.backward()
            optimizer_three.step()
            train_losses.append(train_loss.detach().cpu())

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + backup_name)))
    torch.save(neural_net_three.state_dict(), backup_net_name)

    if plot_result:
        plt.clf()
        plt.plot(np.array(train_losses))
        plt.savefig(graph_name)


def train_three_by_three_for_one_cluster(cluster,
                                         epoch = 1000,
                                         batch_size = 2048,
                                         plot_result = False,
                                         backup_name = "backup_net",
                                         learning_rate = 0.001,
                                         use_pretrained = False,
                                         pretrained_name = "",
                                         training_loss_graph = "training_plots/plot.png"):

    neural_net_three = neural_net_lib.ThreeByThreeSig().to(device)
    params_three = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    train_dataset_three = custom_data_loader_text.CustomDatasetFromTextFiles3(is_small = False,
                                                                              is_clean = False,
                                                                              with_var = True,
                                                                              cluster_num = cluster,
                                                                              device = device)
    train_loader_three = DataLoader(train_dataset_three, **params_three)
    optimizer_three = optim.Adam(neural_net_three.parameters(), lr=learning_rate)
    l1_loss = nn.SmoothL1Loss()
    train_losses = []

    if use_pretrained:
        backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + pretrained_name)))
        neural_net_three.load_state_dict(torch.load(backup_net_name))
        neural_net_three.to(device)


    for e in range (epoch):
        if e%100 == 0:
            print("epoch: "+str(e))
        for i, data in enumerate(train_loader_three):
            inputs, rewards = data
            result = neural_net_three.forward(inputs)
            train_loss = l1_loss(result, rewards)
            optimizer_three.zero_grad()
            train_loss.backward()
            optimizer_three.step()
            train_losses.append(train_loss.detach().cpu())

    backup_name+=("_cluster_"+str(cluster))
    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + backup_name)))
    torch.save(neural_net_three.state_dict(), backup_net_name)

    if plot_result:
        plt.clf()
        plt.plot(np.array(train_losses))
        plt.savefig(training_loss_graph)


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

    if plot_result:
        plt.clf()
        plt.plot(np.array(train_losses))
        plt.savefig(backup_name + ".png")


def train_five_by_five_conv(epoch=1000,
                            batch_size=8192,
                            plot_result=False,
                            backup_name="raw_net_five_conv",
                            learning_rate=0.001,
                            use_pretrained=False,
                            pretrained_name="raw_net_five_conv",
                            training_loss_graph="training_plots/plot.png"):
    print("training convolution net 5 by 5")
    neural_net_five_conv = neural_net_lib.FiveByFiveConv().to(device)

    if use_pretrained:
        backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + pretrained_name)))
        neural_net_five_conv.load_state_dict(torch.load(backup_net_name))
        neural_net_five_conv.to(device)

    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    train_dataset_five = custom_data_loader_text.CustomDatasetFromTextFiles5(is_small=False)
    train_loader_five = DataLoader(train_dataset_five, **params)
    optimizer_five = optim.Adam(neural_net_five_conv.parameters(), lr=learning_rate)
    l1_loss = nn.SmoothL1Loss()
    train_losses = []
    start_time = time.time()
    for e in range(epoch):
        if e%10 == 0:
            end_time = time.time()
            print("epoch: " + str(e))
            print(end_time - start_time)
            start_time = end_time
        for i, data in enumerate(train_loader_five):
            inputs, rewards = data
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
    if plot_result:
        plt.clf()
        plt.plot(np.array(train_losses))
        ax = plt.gca()
        ax.set_ylim([0, 12])
        plt.savefig(training_loss_graph)


def train_five_by_five_for_one_cluster(cluster,
                                       epoch = 1000,
                                       batch_size = 2048,
                                       plot_result = False,
                                       backup_name = "backup_net_five",
                                       learning_rate = 0.001,
                                       use_pretrained = False,
                                       pretrained_name = "raw_net_five_conv",
                                       training_loss_graph="training_plots/plot.png"):
    neural_net_five_conv = neural_net_lib.FiveByFiveConv().to(device)
    if use_pretrained:
        backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + pretrained_name)))
        neural_net_five_conv.load_state_dict(torch.load(backup_net_name))
        neural_net_five_conv.to(device)

    params_five = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    train_dataset_five = custom_data_loader_text.CustomDatasetFromTextFiles5(is_small = False,
                                                                             is_clean = False,
                                                                             with_var = True,
                                                                             cluster_num = cluster,
                                                                             device = device)
    train_loader_five = DataLoader(train_dataset_five, **params_five)
    optimizer_five = optim.Adam(neural_net_five_conv.parameters(), lr=learning_rate)
    l1_loss = nn.SmoothL1Loss()
    train_losses = []

    for e in range (epoch):
        if e%100 == 0:
            print("epoch: "+str(e))
        for i, data in enumerate(train_loader_five):
            inputs, rewards = data
            input_len = len(inputs)
            inputs = inputs.reshape([input_len, 5, 5]).to(device)
            inputs = inputs.unsqueeze(1)
            result = neural_net_five_conv.forward(inputs)
            train_loss = l1_loss(result, rewards)
            optimizer_five.zero_grad()
            train_loss.backward()
            optimizer_five.step()
            train_losses.append(train_loss.detach().cpu())

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(), ("../saved_nets/" + backup_name)))
    torch.save(neural_net_five_conv.state_dict(), backup_net_name)

    if plot_result:
        plt.clf()
        plt.plot(np.array(train_losses))
        plt.savefig(training_loss_graph)

# initialize everything
device = tools.get_device()
print("train_networks uses following cuda device: "+str(device))

