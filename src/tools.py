#!/usr/bin/env python3
# Tim Marvel
import numpy as np
import pyautogui
from time import sleep
import os
import uuid


def move_and_click(w, h):
    pyautogui.moveTo(w, h)
    pyautogui.click()
    sleep(0.02)


def extend_state(state):
    print("called extend state ")
    extended = -1 * np.ones((10, 10))
    extended[1:9, 1:9] = state
    return extended


def rotate_by_90(state):
    rot = np.rot90(state, k=1, axes=(1, 0))
    return rot


def grab_sub_state(state, i, j):
    sub_state = extend_state(state)
    sub_state = sub_state[[i - 1, i, i + 1], :][:, [j - 1, j, j + 1]]
    return sub_state


def grab_sub_state_noext(state, i, j):
    sub_state = state[[i - 1, i, i + 1], :][:, [j - 1, j, j + 1]]
    return sub_state


def move_and_click(w, h):
    pyautogui.moveTo(w, h)
    pyautogui.click()
    sleep(0.02)


def save_action(reward, before, is_test_set):
    hashname = str(uuid.uuid4().hex)
    filename = os.path.join(_pos_location, hashname)
    if is_test_set:
        filename = os.path.join(_pos_location_test, hashname)
    filename_bef = filename + "_before.npy"
    np.save(filename_bef, before)
    filename_reward = filename + "_reward.npy"
    np.save(filename_reward, reward)


def save_action_neg(reward, before, is_test_set):
    hashname = str(uuid.uuid4().hex)
    filename = os.path.join(_neg_location, hashname)
    if is_test_set:
        filename = os.path.join(_neg_location_test, hashname)
    filename_bef = filename + "_before.npy"
    np.save(filename_bef, before)
    filename_reward = filename + "_reward.npy"
    np.save(filename_reward, reward)


def get_save_location():
    return _pos_location, _neg_location


def get_save_test_location():
    return _pos_location_test, _neg_location_test


def start_minesweeper_game():
    # start minesweeper program
    move_and_click(33, 763)
    # can be slow
    sleep(2)
    # start 8 by 8 minesweeper
    move_and_click(739, 320)
    # init
    pyautogui.click()


real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)

# location for the testing set is fixed within the root directory
_pos_location = os.path.join(dir_path, "../collected_data/positive_reward")
_neg_location = os.path.join(dir_path, "../collected_data/negative_reward")
_neg_location_test = os.path.join(dir_path, "../collected_data/test_negative_reward")
_pos_location_test = os.path.join(dir_path, "../collected_data/test_positive_reward")
