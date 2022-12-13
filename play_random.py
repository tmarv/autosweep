#!/usr/bin/env python3
# Tim Marvel
import os
import torch
import random
import math
import json
import numpy as np
from time import sleep

# self made classes
from src import data_gathering_histgrm as dg
from src import reward_manager
from src import minesweeper_interface as min_int
from src import tools

random_percent = 0.0
VERBOSE = False
NUM_ACTIONS = 64
device = 'cpu'

def select_action():
    random_action = random.randrange(NUM_ACTIONS)
    return torch.tensor([[math.floor(random_action / 8), random_action % 8]], device=device, dtype=torch.int)

def play_random(iterations=1, save_data=True):
    print("playing randomly")
    is_test_set = False
    i_episode = 0
    winners = 0
    losers = 0
    while i_episode < iterations:
        sleep(0.3)
        #print("this is i_episode " + str(i_episode))
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
            counter += 1
            for k in range(0, 64):
                action = select_action()
                min_int.move_and_click_to_ij(action[0][0], action[0][1])
                tools.move_to(1490, 900)
                sleep(0.5)
                state = dg.get_state_from_screen()
                if not dg.get_status():
                    state = min_int.mark_game(state)
                sub_state_three = tools.grab_sub_state_three(previous_state, action[0][1] + 1, action[0][0] + 1)
                sub_state_five = tools.grab_sub_state_five(previous_state, action[0][1] + 2, action[0][0] + 2)
                # we hit a mine
                if dg.get_status():
                    losers+=1
                    if save_data:
                        tools.save_action_neg_three(-64, sub_state_three, is_test_set)
                        tools.save_action_neg_five(-64, sub_state_five, is_test_set)
                    tools.move_and_click(1490, 900)
                    counter += 1
                    break

                # compute reward
                reward, has_won = reward_manager.compute_reward(previous_state, state)

                if has_won:
                    winners+=1
                    if save_data:
                        tools.save_action_three(10, sub_state_three, is_test_set)
                        tools.save_action_five(10, sub_state_five, is_test_set)
                    tools.move_and_click(1490, 900)
                    counter += 1001
                    break
                else:
                    if save_data:
                        tools.save_action_three(reward, sub_state_three, is_test_set)
                        tools.save_action_five(reward, sub_state_five, is_test_set)

                previous_state = state.copy()
                state = min_int.mark_game(state)
                # if we didn't act -> k = 100
                if reward != 0:
                    break
                counter += 1

    print("winners: " + str(winners) + " losers: " + str(losers))

def play_the_game(cfg_file_name):
    config_file = open(cfg_file_name)
    config = json.load(config_file)
    # start minesweeper program
    tools.launch_mines()
    # can be slow
    sleep(1)
    # start 8 by 8 minesweeper
    tools.move_and_click(739, 320)
    play_random(config['iterations'], config['save_data'])
    config_file.close()


cfg_file_name = "config/play_random_100.json"
play_the_game(cfg_file_name)