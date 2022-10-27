#!/usr/bin/env python3
# Tim Marvel
import os
import torch
import random
import math
import numpy as np
from time import sleep

# self made classes
from src import data_gathering_histgrm as dg
from src import reward_manager
from src import minesweeper_interface as min_int
#from src import neural_net_lib
from src import tools

random_percent = 0.0
VERBOSE = False
NUM_ACTIONS = 64
device = 'cpu'

def select_action():
    random_action = random.randrange(NUM_ACTIONS)
    # print('this is random action ', random_action)
    return torch.tensor([[math.floor(random_action / 8), random_action % 8]], device=device, dtype=torch.int)

def play_random(iterations=1):
    print("playing with clusters")
    is_test_set = False
    i_episode = 0
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
            #action = select_action_cluster(nets_clusters, state)
            counter += 1
            for k in range(0, 64):
                action = select_action()
                # k1 = random.randrange(64)
                # print(action)
                # print(action[0])
                # print(action[1])
                print("loop k: " + str(dg.get_status()))
                min_int.move_and_click_to_ij(action[0][0], action[0][1])
                # exit()
                # print(action[k])
                tools.move_to(1490, 900)
                # exit()
                # if hit a mine
                sleep(0.5)
                state = dg.get_state_from_screen()
                if not dg.get_status():
                    state = min_int.mark_game(state)
                sub_state_three = tools.grab_sub_state_three(previous_state, action[0][1] + 1, action[0][0] + 1)
                sub_state_five = tools.grab_sub_state_five(previous_state, action[0][1] + 2, action[0][0] + 2)
                #print(sub_state_three)
                #print(sub_state_five)
                #exit()
                # we hit a mine
                if dg.get_status():
                    print('hit mine')
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
                    print("has won with randmo")
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
                    if reward > 0:
                        tools.save_action_three(reward, sub_state_three, is_test_set)
                        tools.save_action_five(reward, sub_state_five, is_test_set)

                previous_state = state.copy()
                state = min_int.mark_game(state)
                # if we didn't act -> k = 100
                if reward != 0:
                    break
                print("updating state")
                counter += 1


# start minesweeper program
tools.launch_mines()
#tools.move_and_click(33, 763)
# can be slow
sleep(1)
# start 8 by 8 minesweeper
tools.move_and_click(739, 320)
play_random(50)