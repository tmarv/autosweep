#!/usr/bin/env python3
# Tim Marvel
import os
import torch
import random
import numpy as np
from time import sleep

# self made classes
from src import data_gathering_histgrm as dg
from src import reward_manager
from src import minesweeper_interface as min_int
from src import neural_net_lib
from src import tools

random_percent = 0

def play_with_three(how_many, epoch, steps):
    neural_net = neural_net_lib.ThreeByThreeSig()
    net_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/neural_net_' + str(epoch) + '_' + str(
            steps)))
    neural_net.load_state_dict(torch.load(net_name))

    i_episode = 0
    while i_episode < how_many:
        i_episode += 1

def select_action_fused(neural_net_three, neural_net_five, state):
    return 0


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
            score_board[i - 2, j - 2] = neural_net.forward(local_tensor)

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


def play_with_nets(how_many, epoch, is_test_set=False):
    print("starting only five by five play")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # load the 5 by 5 kernel network
    neural_net_five = neural_net_lib.FiveByFiveSig()
    net_name_five = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/neural_net_five_' + str(epoch)))
    neural_net_five.load_state_dict(torch.load(net_name_five, map_location=device))
    '''
    # load the 3 by 3 kernel network
    neural_net_three = neural_net_lib.ThreeByThreeSig()
    # TODO: fix this
    net_name_three = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/neural_net_' + str(epoch)))
    neural_net_three.load_state_dict(torch.load(net_name_three)).to(device)
    '''

    i_episode = 0
    while i_episode < how_many:
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
            counter += 1
            for k in range(0, 64):
                print("loop k: "+str(dg.get_status()))
                if random.random() < random_percent:
                    print("random action")
                    action[k][0] = random.randint(0, 7)
                    action[k][1] = random.randint(0, 7)

                min_int.move_and_click_to_ij(action[k][0], action[k][1])
                print(action[k])
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
                    tools.save_action_neg_three(-10, sub_state_three, is_test_set)
                    tools.save_action_neg_five(-10, sub_state_five, is_test_set)
                    print(sub_state_three)
                    print(sub_state_five)
                    tools.move_and_click(1490, 900)
                    counter += 1
                    i_episode = i_episode + 1
                    print("-- should restart")
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

                print("updating state")
                previous_state = state.copy()
                state = min_int.mark_game(state)
                counter += 1



