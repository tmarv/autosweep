#!/usr/bin/env python3
# Tim Marvel
import os
import torch
import random
import math
import json
import numpy as np
from time import sleep
from datetime import datetime

# self made classes
from src import data_gathering_histgrm as dg
from src import reward_manager
from src import minesweeper_interface as min_int
from src import neural_net_lib
from src import tools

#random_percent = 0.0
VERBOSE = False
NUM_ACTIONS = 64


def prepare_return_values(score_board):
    flat = score_board.flatten()
    flat.sort()
    flat = np.flipud(flat)
    return_values = []
    total_len = len(flat)
    i = 0
    while i < total_len:
        local_range = np.where(score_board == flat[i])
        local_sz = len(local_range[0])
        for j in range(0, local_sz):
            return_values.append([local_range[1][j], local_range[0][j]])
            i = i + 1

    if len(return_values) != 64:
        print("Catastrophic error: return values size is off " + str(len(return_values)))
        exit()
    return return_values


def select_action_five(neural_net, state):
    state = tools.extend_state_five(state)
    score_board = np.zeros((8, 8))
    for i in range(2, 10):
        for j in range(2, 10):
            # todo adapt this
            local_cpy = tools.grab_sub_state_noext_five(state, j, i)
            local_tensor = torch.from_numpy(local_cpy).to(dtype=torch.float)
            local_tensor = local_tensor.unsqueeze(0)
            score_board[i - 2, j - 2] = neural_net.forward(local_tensor.reshape([1, 25]))

    flat = score_board.flatten()
    flat.sort()
    flat = np.flipud(flat)
    return_values = []
    total_len = len(flat)
    i = 0
    while i < total_len:
        local_range = np.where(score_board == flat[i])
        local_sz = len(local_range[0])
        for j in range(0, local_sz):
            return_values.append([local_range[0][j], local_range[1][j]])
            i = i + 1
    if len(return_values) != 64:
        print("Catastrophic error: return values size is off " + str(len(return_values)))
        exit()
    return return_values


def run_clustering_five(state):
    state_five = tools.extend_state_five(state)
    score_board_clusters = np.zeros((8, 8, 3))
    for i in range(2, 10):
        for j in range(2, 10):
            local_cpy = tools.grab_sub_state_noext_five(state_five, i, j)
            local_tensor = torch.from_numpy(local_cpy).to(dtype=torch.float)
            local_tensor = local_tensor.reshape([1, 5, 5])
            local_tensor = local_tensor.unsqueeze(0)
            local_tensor = local_tensor.to(device)
            float_result = cluster_net.forward(local_tensor)
            score_board_clusters[i - 2, j - 2, :] = float_result[0].cpu().detach().numpy()
    return score_board_clusters


def select_action_cluster(the_nets, state):
    clusters = run_clustering_five(state)
    state_five = tools.extend_state_five(state)
    score_board = np.zeros((8, 8))

    for i in range(2, 10):
        for j in range(2, 10):
            local_cpy = tools.grab_sub_state_noext_five(state_five, i, j)
            local_tensor = torch.from_numpy(local_cpy).to(dtype=torch.float)
            local_tensor = local_tensor.reshape([1, 5, 5])
            local_tensor = local_tensor.unsqueeze(0)
            local_tensor = local_tensor.to(device)
            cluster = clusters[i - 2, j - 2]
            mult0 = mult0_conf
            mult1 = mult1_conf
            mult2 = mult2_conf

            if cluster[0] < 0.05:
                mult0 = 0
            if cluster[1] < 0.4:
                mult1 = 0
            if cluster[2] < 0.2:
                mult2 = 0
            result0 = the_nets[0].forward(local_tensor)[0]
            result1 = the_nets[1].forward(local_tensor)[0]
            result2 = the_nets[2].forward(local_tensor)[0]
            score_board[i - 2, j - 2] = result0 * mult0 + (result1 * mult1) - abs(result2 * mult2)
    return prepare_return_values(score_board)


def init_the_cluster_nets_five(base_name):

    net_0_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/' + base_name + "0"))
    net_1_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/' + base_name + "1"))
    net_2_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/' + base_name + "2"))

    net_0 = neural_net_lib.FiveByFiveConv()
    net_0.load_state_dict(torch.load(net_0_name))
    net_0.eval()
    net_0.to(device)

    net_1 = neural_net_lib.FiveByFiveConv()
    net_1.load_state_dict(torch.load(net_1_name))
    net_1.eval()
    net_1.to(device)

    net_2 = neural_net_lib.FiveByFiveConv()
    net_2.load_state_dict(torch.load(net_2_name))
    net_2.eval()
    net_2.to(device)

    return [net_0, net_1, net_2]



def play_with_clustering(nets_clusters, iterations = 1, random_percent=0.0, save_data=True):
    print("playing with clusters on a 5 by 5 grid")
    print(save_data)
    is_test_set = False
    i_episode = 0
    winners = 0
    losers = 0
    while i_episode < iterations:
        sleep(0.3)
        print("this is i_episode " + str(i_episode))
        i_episode = i_episode + 1
        tools.move_and_click(739, 320)
        sleep(0.3)
        state = dg.get_state_from_screen()
        # run the flagging algorithm
        state = min_int.mark_game(state)
        # perform a deep copy
        previous_state = state.copy()
        sleep(0.3)
        counter = 0
        while not dg.get_status() and counter < 200:
            action = select_action_cluster(nets_clusters, state)
            counter += 1
            for k in range(0, 64):
                if random.random() < random_percent:
                    print("random action")
                    action[k][0] = random.randint(0, 7)
                    action[k][1] = random.randint(0, 7)

                min_int.move_and_click_to_ij(action[k][0], action[k][1])
                tools.move_to(1490, 900)
                sleep(0.3)
                state = dg.get_state_from_screen()
                if not dg.get_status():
                    state = min_int.mark_game(state)
                sub_state_three = tools.grab_sub_state_three(previous_state, action[k][1] + 1, action[k][0] + 1)
                sub_state_five = tools.grab_sub_state_five(previous_state, action[k][1] + 2, action[k][0] + 2)

                # we hit a mine
                if dg.get_status():
                    losers += 1
                    if save_data:
                        tools.save_action_neg_three(-64, sub_state_three, is_test_set)
                        tools.save_action_neg_five(-64, sub_state_five, is_test_set)
                    tools.move_and_click(1490, 900)
                    counter += 1
                    break

                # compute reward
                reward, has_won = reward_manager.compute_reward(previous_state, state)

                if has_won:
                    winners += 1
                    if save_data:
                        tools.save_action_three(10, sub_state_three, is_test_set)
                        tools.save_action_five(10, sub_state_five, is_test_set)
                    tools.move_and_click(1490, 900)
                    counter += 1001
                    break
                else:
                    # save data from transition
                    if save_data and reward > 0:
                        tools.save_action_three(reward, sub_state_three, is_test_set)
                        tools.save_action_five(reward, sub_state_five, is_test_set)
                        #print("this is positive reward " + str(reward))

                    elif save_data and reward <= 0:
                        #print("this is negative reward " + str(reward))
                        tools.save_action_neg_three(reward, sub_state_three, is_test_set)
                        tools.save_action_neg_five(reward, sub_state_five, is_test_set)

                previous_state = state.copy()
                state = min_int.mark_game(state)
                # if we didn't act -> k = 100
                if reward != 0:
                    break
                #print("updating state")
                counter += 1

    print("winners: " + str(winners) + " losers: " + str(losers))

#load the config

cfg_file_name = "config/play_5_convolutional.json"
config_file = open(cfg_file_name)
config = json.load(config_file)

# prepare and load the nets
device = tools.get_device()

cluster_net = neural_net_lib.FiveByFiveConvCluster()

cluster_net_name = os.path.abspath(
    os.path.join(tools.get_working_dir(), '../saved_nets/'+config["name_of_net_cluster_net"]))
cluster_net.load_state_dict(torch.load(cluster_net_name))
cluster_net.eval()
cluster_net.to(device)
loaded_nets_clusters = init_the_cluster_nets_five(config["base_name_of_individual_cluster_nets"])

mult0_conf = config["mult_factor_0"]
mult1_conf = config["mult_factor_1"]
mult2_conf = config["mult_factor_2"]

# start minesweeper program
tools.launch_mines()
# can be slow
sleep(1)
# start 8 by 8 minesweeper
tools.move_and_click(739, 320)

torch.set_num_threads(4)
with torch.no_grad():
    play_with_clustering(loaded_nets_clusters,
                         iterations = config["iterations"],
                         random_percent = config["random_percent"],
                         save_data = config["save_data"])
