#!/usr/bin/env python3
# Tim Marvel
import math
from time import sleep
import random
import pyautogui as gui
import torch
# self made classes
from src import data_gathering_histgrm as dg
from src import reward_manager
from src import minesweeper_interface as min_int
from src import tools

#  Global variables
NUM_ACTIONS = 64  # size of an 8 by 8 minefield

#pos_location, neg_location = tools.get_save_location_three()

# no need for a gpu if it is an purely random play
device = torch.device("cpu")

# start minesweeper program
tools.move_and_click(33, 763)
# can be slow
sleep(1)
# start 8 by 8 minesweeper
tools.move_and_click(739, 320)
# init
gui.click()


def select_action():
    random_action = random.randrange(NUM_ACTIONS)
    # print('this is random action ', random_action)
    return torch.tensor([[math.floor(random_action / 8), random_action % 8]], device=device, dtype=torch.int)


total_reward = 0.0
reward_counter = 0


def play_the_game(how_many, is_test_set=False):
    for i_episode in range(how_many):
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
        while not dg.get_status() and counter < 100:
            action = select_action()
            counter += 1
            # print(action)
            # print(action[0])
            # TODO Fix this mess!
            for action_ind in action[0]:
                action_num = action[0][0] * 8 + action[0][1]
                # print('-----')
                # print(action_num)
                ii = math.floor(action_num / 8)
                jj = math.floor(action_num % 8)
                # print('i ', math.floor(action_num / 8))
                # print('j ', action_num % 8)
                min_int.move_and_click_to_ij(action_num % 8, math.floor(action_num / 8))
                gui.moveTo(1490, 900)
                # if hit a mine
                sleep(0.5)
                new_state = dg.get_state_from_screen()
                if dg.get_status():
                    # print('hit mine')
                    sub_state = tools.grab_sub_state_three(state, ii + 1, jj + 1)
                    tools.save_action_neg_three(-10, sub_state, is_test_set)
                    # print(sub_state)
                    tools.move_and_click(1490, 900)
                    counter += 1
                    break

                # compute reward
                reward, has_won = reward_manager.compute_reward(state, new_state)
                sub_state = tools.grab_sub_state_three(state, ii + 1, jj + 1)
                if not has_won:
                    # print("no win")
                    # save data from transition
                    if reward > 0:
                        tools.save_action_three(reward, sub_state, is_test_set)
                        # print("reward is positive")

                    elif reward <= 0:
                        tools.save_action_neg_three(reward, sub_state, is_test_set)
                        # print("reward is negative")

                    state = new_state
                    state = min_int.mark_game(state)
                    # print(reward)
                    break

                if has_won:
                    # print("has won")
                    tools.save_action_three(10, sub_state, is_test_set)
                    tools.move_and_click(1490, 900)
                    tools.move_and_click(739, 320)
                    gui.click()
                    break

                state = new_state

                # mark cells
                state = min_int.mark_game(state)
                counter += 1
