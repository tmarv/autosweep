#!/usr/bin/env python3
# Tim Marvel
import numpy as np
import pyautogui
from time import sleep
import os
import uuid
import subprocess
import torch

def move_and_click(w, h):
    pyautogui.moveTo(w, h)
    pyautogui.click()
    sleep(0.02)


def extend_state(state):
    extended = -2 * np.ones((10, 10))
    extended[1:9, 1:9] = state
    return extended


def extend_state_five(state):
    extended_five = -2 * np.ones((12, 12))
    extended_five[2:10, 2:10] = state
    return extended_five


def extend_state_seven(state):
    extended_seven = -2 * np.ones((14, 14))
    extended_seven[3:11, 3:11] = state
    return extended_seven    


def rotate_by_90(state):
    rot = np.rot90(state, k=1, axes=(1, 0))
    return rot


def rotate_by_90_flat(state):
    rot = np.rot90(state.reshape(3,3), k=1, axes=(1, 0))
    return rot.flatten()


def rotate_by_90_flat_five(state):
    rot = np.rot90(state.reshape(5, 5), k=1, axes=(1, 0))
    return rot.flatten()


def grab_sub_state_three(state, i, j):
    sub_state = extend_state(state)
    sub_state = sub_state[[i - 1, i, i + 1], :][:, [j - 1, j, j + 1]]
    return sub_state


def grab_sub_state_five(state, i, j):
    sub_state = extend_state_five(state)
    sub_state = sub_state[[i - 2, i - 1, i, i + 1, i + 2], :][:, [j - 2, j - 1, j, j + 1, j + 2]]
    return sub_state


def grab_sub_state_seven(state, i, j):
    sub_state = extend_state_seven(state)
    sub_state = sub_state[[i - 3, i - 2, i - 1, i, i + 1, i + 2, i + 3], :][:, [j - 3, j - 2, j - 1, j, j + 1, j + 2, j + 3]]
    return sub_state


def grab_sub_state_noext(state, i, j):
    sub_state = state[[i - 1, i, i + 1], :][:, [j - 1, j, j + 1]]
    return sub_state


def grab_sub_state_noext_five(state, i, j):
    sub_state = state[[i - 2, i - 1, i, i +1, i + 2], :][:, [j - 2, j-1, j, j + 1, j + 2]]
    return sub_state


def augment_data(rewards, data_points):
    aug_rewards = []
    aug_data_pts = []
    for i in range(len(rewards)):
        state = data_points[i]
        reward = rewards[i]
        state_b = rotate_by_90_flat(state)
        state_c = rotate_by_90_flat(state_b)
        state_d = rotate_by_90_flat(state_c)
        aug_rewards.append(reward)
        aug_rewards.append(reward)
        aug_rewards.append(reward)
        aug_rewards.append(reward)
        aug_data_pts.append(state)
        aug_data_pts.append(state_b)
        aug_data_pts.append(state_c)
        aug_data_pts.append(state_d)
    print(len(aug_rewards))
    print(len(aug_data_pts))
    return np.array(aug_rewards), np.array(aug_data_pts)

def augment_data_five(rewards, data_points):
    aug_rewards = []
    aug_data_pts = []
    for i in range(len(rewards)):
        state = data_points[i]
        reward = rewards[i]
        state_b = rotate_by_90_flat_five(state)
        state_c = rotate_by_90_flat_five(state_b)
        state_d = rotate_by_90_flat_five(state_c)
        aug_rewards.append(reward)
        aug_rewards.append(reward)
        aug_rewards.append(reward)
        aug_rewards.append(reward)
        aug_data_pts.append(state)
        aug_data_pts.append(state_b)
        aug_data_pts.append(state_c)
        aug_data_pts.append(state_d)

    return np.array(aug_rewards), np.array(aug_data_pts)


### Todo remove duplicate
def move_to(w, h):
    pyautogui.moveTo(w, h)
    sleep(0.02)


def save_action_text_three(reward, before):
    # turn the grid into a csv
    list = before.flatten().tolist()
    list = ','.join(str(v) for v in list)
    _rewards3_text_file.write(list)
    _rewards3_text_file.write(','+str(reward))
    _rewards3_text_file.write('\n')


def save_action_text_five(reward, before):
    # turn the grid into a csv
    list = before.flatten().tolist()
    list = ','.join(str(v) for v in list)
    _rewards5_text_file.write(list)
    _rewards5_text_file.write(','+str(reward))
    _rewards5_text_file.write('\n')


def save_action_text_seven(reward, before):
    # turn the grid into a csv
    list = before.flatten().tolist()
    list = ','.join(str(v) for v in list)
    _rewards7_text_file.write(list)
    _rewards7_text_file.write(','+str(reward))
    _rewards7_text_file.write('\n')


def save_action_three(reward, before, is_test_set):
    save_action_text_three(reward, before)

def save_action_five(reward, before, is_test_set):
    save_action_text_five(reward, before)

def save_action_seven(reward, before, is_test_set):
    save_action_text_seven(reward, before)

def save_action_neg_three(reward, before, is_test_set):
    save_action_text_three(reward, before)

def save_action_neg_five(reward, before, is_test_set):
    save_action_text_five(reward, before)

def save_action_neg_seven(reward, before, is_test_set):
    save_action_text_seven(reward, before)



def get_device():
    return device


def start_minesweeper_game():
    # start minesweeper program
    move_and_click(33, 763)
    # can be slow
    sleep(2)
    # start 8 by 8 minesweeper
    move_and_click(739, 320)
    # init
    pyautogui.click()


def launch_mines():
    # start the game directly
    start_game = "/usr/games/gnome-mines"
    res = subprocess.Popen(start_game, shell=False)
    # os.system(start_game)

def get_working_dir():
    return dir_path


def get_text_file_names():
    return [_data_pts_3_filename, _data_pts_5_filename]


def get_text_file_names_small():
    # just to try it out
    #return [_data_pts_3_filename_var, _data_pts_5_filename_small]
    return [_data_pts_3_filename_small, _data_pts_5_filename_small]


def get_text_file_names_var():
    return [_data_pts_3_filename_var, _data_pts_5_filename_var]

def get_text_file_names_clean():
    return [_data_pts_3_filename_clean, _data_pts_5_filename_clean]


real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_data_pts_3_filename_small = os.path.join(dir_path, "../collected_data/rewards3_short.txt")
_data_pts_3_filename = os.path.join(dir_path, "../collected_data/rewards3.txt")
_data_pts_3_filename_var = os.path.join(dir_path, "../collected_data/rewards3_var.txt")
_data_pts_3_filename_clean = os.path.join(dir_path, "../collected_data/rewards3_clean.txt")

_data_pts_5_filename_small = os.path.join(dir_path, "../collected_data/rewards5_short.txt")
_data_pts_5_filename = os.path.join(dir_path, "../collected_data/rewards5.txt")
_data_pts_5_filename_var = os.path.join(dir_path, "../collected_data/rewards5_var.txt")
_data_pts_5_filename_clean = os.path.join(dir_path, "../collected_data/rewards5_clean.txt")

_data_pts_7_filename = os.path.join(dir_path, "../collected_data/rewards7.txt")

_rewards3_text_file = open(_data_pts_3_filename, 'a')
_rewards5_text_file = open(_data_pts_5_filename, 'a')
_rewards7_text_file = open(_data_pts_7_filename, 'a')

def __del__(self):
    _rewards3_text_file.close()
    _rewards5_text_file.close()
    _rewards7_text_file.close()


'''
just making sure the five by five works
mytest = np.array([1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5])
my_result = rotate_by_90_flat_five(mytest)
print(my_result)
'''