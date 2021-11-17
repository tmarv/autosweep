#!/usr/bin/env python3
# Tim Marvel
import math
import os
from time import sleep
import random

import numpy as np
import pyautogui as gui
import torch
# self made classes
from src import data_gathering_histgrm as dg
from src import reward_manager
from src import minesweeper_interface as min_int
from src import neural_net_lib
from src import tools

#  Global variables


NUM_ACTIONS = 64  # size of an 8 by 8 minefield

pos_location, neg_location = tools.get_save_location_three()

# TODO check why action is in state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# start minesweeper program
tools.move_and_click(33, 763)
# can be slow
sleep(1)
# start 8 by 8 minesweeper
tools.move_and_click(739, 320)
# init
gui.click()

total_reward = 0.0
reward_counter = 0


def select_action_three(neural_net, state):
    state = tools.extend_state(state)
    # print("state")
    # print(state)
    score_board = np.zeros((8, 8))
    for i in range(1, 9):
        for j in range(1, 9):
            local_cpy = tools.grab_sub_state_noext(state, j, i)
            local_tensor = torch.from_numpy(local_cpy).to(dtype=torch.float)
            local_tensor = local_tensor.unsqueeze(0)
            score_board[i - 1, j - 1] = neural_net.forward(local_tensor)

    # print("score board " + str(score_board))
    flat = score_board.flatten()
    flat.sort()
    flat = np.flipud(flat)
    return_values = []
    total_len = len(flat)
    # print("this is total len      "+str(total_len))
    print("this is flat: " + str(flat))
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


def play_the_game_and_collect(how_many, epoch, steps, is_test_set=False, random_percent=0.0):
    net_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/neural_net_' + str(epoch) + '_' + str(
            steps)))
    print("path: " + str(net_name))
    neural_net = neural_net_lib.ThreeByThreeSig()
    neural_net.load_state_dict(torch.load(net_name))
    neural_net.eval()
    i_episode = 0
    while i_episode < how_many:
        # for i_episode in range(how_many):
        # click on a start location
        sleep(0.3)
        print("this is i_episode " + str(i_episode))
        tools.move_and_click(739, 320)
        sleep(0.3)
        state = dg.get_state_from_screen()
        # run the flagging algorithm
        state = min_int.mark_game(state)
        sleep(0.3)
        counter = 0
        has_won = False
        while not dg.get_status() and counter < 1000:
            action = select_action_three(neural_net, state)
            # print(action)
            counter += 1
            for k in range(0, 64):
                print("this is k " + str(k))
                print(action[k][0], action[k][1])
                # generate random num
                if random.random() < random_percent:
                    print("random action")
                    action[k][0] = random.randint(0, 7)
                    action[k][1] = random.randint(0, 7)
                # else:
                print(action[k][0], action[k][1])
                min_int.move_and_click_to_ij(action[k][0], action[k][1])
                gui.moveTo(1490, 900)
                # if hit a mine
                sleep(0.5)
                new_state = dg.get_state_from_screen()
                sub_state_three = tools.grab_sub_state_three(state, action[k][1] + 1, action[k][0] + 1)
                sub_state_five = tools.grab_sub_state_five(state, action[k][1] + 2, action[k][0] + 2)
                if dg.get_status():
                    print('hit mine')
                    tools.save_action_neg_three(-10, sub_state_three, is_test_set)
                    tools.save_action_neg_five(-10, sub_state_five, is_test_set)
                    print(sub_state_three)
                    print(sub_state_five)
                    tools.move_and_click(1490, 900)
                    counter += 1
                    print("DEBUG 1")
                    i_episode = i_episode + 1
                    break

                # compute reward
                reward, has_won = reward_manager.compute_reward(state, new_state)
                # sub_state = tools.grab_sub_state_three(state, action[k][1] + 1, action[k][0] + 1)
                # print("reward " + str(reward))
                # print(sub_state)
                if not has_won:
                    # print("no win")
                    # save data from transition
                    if reward > 0:
                        tools.save_action_three(reward, sub_state_three, is_test_set)
                        tools.save_action_five(reward, sub_state_five, is_test_set)
                        print("this is reward " + str(reward))
                        state = new_state
                        state = min_int.mark_game(state)
                        break

                    elif reward <= 0:
                        print("this is reward " + str(reward))
                        tools.save_action_neg_three(reward, sub_state_three, is_test_set)
                        tools.save_action_neg_five(reward, sub_state_five, is_test_set)

                    state = new_state
                    state = min_int.mark_game(state)
                    counter += 1
                    continue

                if has_won:
                    print("has won collect data with ml")
                    tools.save_action_three(10, sub_state_three, is_test_set)
                    tools.save_action_five(10, sub_state_five, is_test_set)
                    tools.move_and_click(1490, 900)
                    # tools.move_and_click(739, 320)
                    # gui.click()
                    state = new_state
                    state = min_int.mark_game(state)
                    print("has won collect data with ml :" + str(counter))
                    counter += 1001
                    print("has won collect data with ml :" + str(i_episode))
                    print("has won collect data with ml :" + str(counter))
                    i_episode = i_episode + 1
                    # print("DEBUG 3")
                    break

                state = new_state
                print("incrementing i episode" + str(i_episode))
                i_episode += 1
                # mark cells
                state = min_int.mark_game(state)
                counter += 1
