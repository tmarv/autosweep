#!/usr/bin/env python3
# Tim Marvel
import os
import torch
import random
import math
import numpy as np
from time import sleep
from datetime import datetime

# self made classes
from src import data_gathering_histgrm as dg
from src import reward_manager
from src import minesweeper_interface as min_int
from src import neural_net_lib
from src import tools

random_percent = 0.0
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
        # print("this is i at start: "+str(i))
        local_range = np.where(score_board == flat[i])
        # print(local_range)
        # print("this is flat[i]: "+str(flat[i]))
        # print("this is local range: "+str(local_range))
        local_sz = len(local_range[0])
        for j in range(0, local_sz):
            return_values.append([local_range[1][j], local_range[0][j]])
            # print("str: "+str([local_range[0][j], local_range[1][j]]))
            i = i + 1
            # print("this is i after: " + str(i))
        # if local_sz > 1:
        # print("----------- local_sz is bigger than 1")
        # print(local_sz)
        # print(flat)
        # exit()

    if len(return_values) != 64:
        print("Catastrophic error: return values size is off " + str(len(return_values)))
        exit()
    # print("this is size:    "+str(len(return_values)))
    # print(return_values)
    # print(flat)
    return return_values


def select_action_five(neural_net, state):
    state = tools.extend_state_five(state)
    # print("state")
    # print(state)
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
        # print("this is i at start: "+str(i))
        local_range = np.where(score_board == flat[i])
        # print("this is local range: "+str(local_range))
        local_sz = len(local_range[0])
        for j in range(0, local_sz):
            return_values.append([local_range[0][j], local_range[1][j]])
            # print("str: "+str([local_range[0][j], local_range[1][j]]))
            i = i + 1
            # print("this is i after: " + str(i))
    if len(return_values) != 64:
        print("Catastrophic error: return values size is off " + str(len(return_values)))
        exit()
    # print("this is size:    "+str(len(return_values)))
    return return_values


def run_clustering_five(state):
    # print("state: "+str(state))
    state_five = tools.extend_state_five(state)
    #print("state_five: "+str(state_five))
    score_board_clusters = np.zeros((8, 8, 3))
    for i in range(2, 10):
        for j in range(2, 10):
            local_cpy = tools.grab_sub_state_noext_five(state_five, i, j)
            #print("local_cpy: " + str(local_cpy))
            local_tensor = torch.from_numpy(local_cpy).to(dtype=torch.float)
            #local_tensor = local_tensor.unsqueeze(0)
            local_tensor = local_tensor.reshape([1, 5, 5])
            local_tensor = local_tensor.unsqueeze(0)
            #print('local_tensor'+str(local_tensor))
            float_result = cluster_net.forward(local_tensor)
            # print(local_cpy)
            # print(float_result[0].detach().numpy())
            # score_board_clusters[j - 1, i - 1] = round(torch.argmax(float_result).item())
            score_board_clusters[i - 2, j - 2, :] = float_result[0].detach().numpy()
    return score_board_clusters


def select_action_cluster(the_nets, state):
    # clusters = run_clustering_three(state)
    print("before run clustering: " + str(datetime.now()))
    clusters = run_clustering_five(state)
    print("after run clustering: " + str(datetime.now()))
    # print(state)
    state_five = tools.extend_state_five(state)
    score_board = np.zeros((8, 8))

    for i in range(2, 10):
        for j in range(2, 10):
            local_cpy = tools.grab_sub_state_noext_five(state_five, i, j)
            local_tensor = torch.from_numpy(local_cpy).to(dtype=torch.float)
            local_tensor = local_tensor.reshape([1, 5, 5])
            local_tensor = local_tensor.unsqueeze(0)
            cluster = clusters[i - 2, j - 2]
            # print("cluster alone")
            # print(cluster)
            # print(local_cpy)
            # if(j==2):
            #    exit()
            mult0 = 5.0
            mult1 = 0.5
            mult2 = 1.2
            # print(cluster[0])
            # print(cluster[1])
            # print(cluster[2])
            if cluster[0] < 0.1:
                mult0 = 0
            if cluster[1] < 0.1:
                mult1 = 0
            if cluster[2] < 0.1:
                mult2 = 0
            result0 = the_nets[0].forward(local_tensor)[0]
            result1 = the_nets[1].forward(local_tensor)[0]
            result2 = the_nets[2].forward(local_tensor)[0]
            # print(result0*mult0+(result1*mult1)-abs(result2*mult2))
            score_board[i - 2, j - 2] = result0 * mult0 + (result1 * mult1) - abs(result2 * mult2)
    # print(score_board)
    print("after run algo score board: " + str(datetime.now()))
    return prepare_return_values(score_board)


def init_the_cluster_nets_five(base_name):
    net_0 = neural_net_lib.FiveByFiveConv()
    net_1 = neural_net_lib.FiveByFiveConv()
    net_2 = neural_net_lib.FiveByFiveConv()

    net_0_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/' + base_name + "0"))
    net_1_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/' + base_name + "1"))
    net_2_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/' + base_name + "2"))

    net_0.load_state_dict(torch.load(net_0_name, map_location=device))
    net_1.load_state_dict(torch.load(net_1_name, map_location=device))
    net_2.load_state_dict(torch.load(net_2_name, map_location=device))
    net_0.eval()
    net_1.eval()
    net_2.eval()
    return [net_0, net_1, net_2]


def play_with_nets(iterations, epoch='', is_test_set=False, random_percent=0.0):
    print("starting only five by five play")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # load the 5 by 5 kernel network
    neural_net_five = neural_net_lib.FiveByFiveSig()
    net_name_five = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/raw_net_five'))
    neural_net_five.load_state_dict(torch.load(net_name_five, map_location=device))
    neural_net_five.eval()
    win = 0
    lose = 0
    # load the 3 by 3 kernel network

    # TODO fix incrementation bug
    i_episode = 0
    while i_episode < iterations:
        print("-- restarted at while")
        sleep(0.3)
        print("this is i_episode " + str(i_episode))
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
            print("loop not")
            action = select_action_five(neural_net_five, state)
            # action = select_action_fused(neural_net_three, neural_net_five, state)
            counter += 1
            for k in range(0, 64):
                print("loop k: " + str(dg.get_status()))
                if random.random() < random_percent:
                    print("random action")
                    action[k][0] = random.randint(0, 7)
                    action[k][1] = random.randint(0, 7)

                min_int.move_and_click_to_ij(action[k][0], action[k][1])
                # print(action[k])
                tools.move_to(1490, 900)
                # if hit a mine
                sleep(0.5)
                state = dg.get_state_from_screen()
                if not dg.get_status():
                    state = min_int.mark_game(state)
                sub_state_three = tools.grab_sub_state_three(previous_state, action[k][1] + 1, action[k][0] + 1)
                sub_state_five = tools.grab_sub_state_five(previous_state, action[k][1] + 2, action[k][0] + 2)

                # we hit a mine
                if dg.get_status():
                    print('hit mine')
                    tools.save_action_neg_three(-64, sub_state_three, is_test_set)
                    tools.save_action_neg_five(-64, sub_state_five, is_test_set)
                    print(sub_state_three)
                    print(sub_state_five)
                    tools.move_and_click(1490, 900)
                    counter += 1
                    i_episode = i_episode + 1
                    print("-- should restart")
                    lose += 1
                    break

                # compute reward
                reward, has_won = reward_manager.compute_reward(previous_state, state)

                if has_won:
                    print("has won collect data with ml")
                    tools.save_action_three(10, sub_state_three, is_test_set)
                    tools.save_action_five(10, sub_state_five, is_test_set)
                    tools.move_and_click(1490, 900)
                    print("has won collect data with ml :" + str(counter))
                    counter += 1001
                    i_episode = i_episode + 1
                    print("should restart")
                    win += 1
                    break
                else:
                    print("no win")
                    # save data from transition
                    if reward > 0:
                        tools.save_action_three(reward, sub_state_three, is_test_set)
                        tools.save_action_five(reward, sub_state_five, is_test_set)
                        print("this is positive reward " + str(reward))

                    elif reward <= 0:
                        print("this is negative reward " + str(reward))
                        tools.save_action_neg_three(reward, sub_state_three, is_test_set)
                        tools.save_action_neg_five(reward, sub_state_five, is_test_set)

                previous_state = state.copy()
                state = min_int.mark_game(state)
                # if we didn't act -> k = 100
                if reward != 0:
                    print("call ml again")
                    break
                print("updating state")
                counter += 1
    print("win" + str(win))
    print("lose" + str(lose))


def play_with_clustering(iterations = 1, random_percent=0.0):
    nets_clusters = init_the_cluster_nets_five("net_five_cluster_five_")
    print("playing with clusters")
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
            print("gloabl before: "+str(datetime.now()))
            action = select_action_cluster(nets_clusters, state)
            print("global after: "+str(datetime.now()))
            counter += 1
            for k in range(0, 64):
                print("loop k: " + str(dg.get_status()))
                '''
                if random.random() < random_percent:
                    print("random action")
                    action[k][0] = random.randint(0, 7)
                    action[k][1] = random.randint(0, 7)
                '''
                min_int.move_and_click_to_ij(action[k][0], action[k][1])
                # print(action[k])
                tools.move_to(1490, 900)
                # exit()
                # if hit a mine
                sleep(0.3)
                state = dg.get_state_from_screen()
                if not dg.get_status():
                    state = min_int.mark_game(state)
                sub_state_three = tools.grab_sub_state_three(previous_state, action[k][1] + 1, action[k][0] + 1)
                sub_state_five = tools.grab_sub_state_five(previous_state, action[k][1] + 2, action[k][0] + 2)

                # we hit a mine
                if dg.get_status():
                    print('hit mine')
                    losers += 1
                    tools.save_action_neg_three(-64, sub_state_three, is_test_set)
                    tools.save_action_neg_five(-64, sub_state_five, is_test_set)
                    print(sub_state_three)
                    print(sub_state_five)
                    tools.move_and_click(1490, 900)
                    counter += 1
                    # i_episode = i_episode + 1
                    print("-- should restart")
                    break

                # compute reward
                reward, has_won = reward_manager.compute_reward(previous_state, state)

                if has_won:
                    winners += 1
                    print("has won with clusters")
                    tools.save_action_three(10, sub_state_three, is_test_set)
                    tools.save_action_five(10, sub_state_five, is_test_set)
                    tools.move_and_click(1490, 900)
                    print("has won collect data with ml :" + str(counter))
                    counter += 1001
                    # i_episode = i_episode + 1
                    print("should restart")
                    break
                else:
                    print("no win")
                    # save data from transition
                    if reward > 0:
                        tools.save_action_three(reward, sub_state_three, is_test_set)
                        tools.save_action_five(reward, sub_state_five, is_test_set)
                        print("this is positive reward " + str(reward))

                    elif reward <= 0:
                        print("this is negative reward " + str(reward))
                        tools.save_action_neg_three(reward, sub_state_three, is_test_set)
                        tools.save_action_neg_five(reward, sub_state_five, is_test_set)

                previous_state = state.copy()
                state = min_int.mark_game(state)
                # if we didn't act -> k = 100
                if reward != 0:
                    print("call ml again")
                    break
                print("updating state")
                counter += 1

    print("winners: " + str(winners) + " losers: " + str(losers))






device = tools.get_device()
#device = "cpu"

cluster_net = neural_net_lib.FiveByFiveConvCluster()
cluster_net_name = os.path.abspath(
    os.path.join(tools.get_working_dir(), '../saved_nets/backup_net_cluster_five'))
cluster_net.load_state_dict(torch.load(cluster_net_name, map_location=device))
cluster_net.eval()

# start minesweeper program
tools.launch_mines()
#tools.move_and_click(33, 763)
# can be slow
sleep(1)
# start 8 by 8 minesweeper
tools.move_and_click(739, 320)
# init
# gui.click()

torch.set_num_threads(4)
with torch.no_grad():
    play_with_clustering(iterations=20)
# play_with_nets(iterations=1)
# play_random(iterations=10)
