#!/usr/bin/env python3
# Tim Marvel
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import src.tools
from src import reward_manager, tools


class CustomDataset(Dataset):

    def __init__(self, neg_path_to_folder, pos_path_to_folder):
        self.positive_percent = 0.5

        self.neg_path_to_folder = neg_path_to_folder
        self.all_neg_elements_in_dir_list = os.listdir(neg_path_to_folder)
        self.all_neg_elements_in_dir_list.sort()
        self.neg_size = math.floor(len(self.all_neg_elements_in_dir_list) / 2)

        self.pos_path_to_folder = pos_path_to_folder
        self.all_pos_elements_in_dir_list = os.listdir(pos_path_to_folder)
        self.all_pos_elements_in_dir_list.sort()
        self.pos_size = math.floor(len(self.all_pos_elements_in_dir_list) / 2)

        self.amount_of_training_pts = (self.neg_size + self.pos_size)

    # index isn't actually used -> we want to control how many negative or positive samples we train
    def __getitem__(self, index):
        toggle = random.random()
        # print("** called")
        if toggle < self.positive_percent:
            r = random.randint(0, self.neg_size - 1)
            reward = np.load((self.neg_path_to_folder + "/" + self.all_neg_elements_in_dir_list[2 * r + 2]))
            original = np.load((self.neg_path_to_folder + "/" + self.all_neg_elements_in_dir_list[2 * r + 1]))
            reward = reward_manager.reward_shaper_neg_three(reward, original)
            rot1 = tools.rotate_by_90(original)
            rot2 = tools.rotate_by_90(rot1)
            rot3 = tools.rotate_by_90(rot2)
            # print("debug 1")
            ret_rewards = torch.Tensor(np.array([reward, reward, reward, reward]))
            ret_inputs = torch.Tensor(np.array([original, rot1, rot2, rot3]))
            return ret_inputs, ret_rewards
        else:
            # print("----- called else")
            r = random.randint(0, self.pos_size - 1)
            reward = np.load((self.pos_path_to_folder + "/" + self.all_pos_elements_in_dir_list[2 * r + 2]))
            original = np.load((self.pos_path_to_folder + "/" + self.all_pos_elements_in_dir_list[2 * r + 1]))
            reward = reward_manager.reward_shaper_pos_three(reward, original)
            rot1 = tools.rotate_by_90(original)
            rot2 = tools.rotate_by_90(rot1)
            rot3 = tools.rotate_by_90(rot2)
            # print(reward)
            # dummy1 = np.array([reward, reward, reward, reward])
            ret_rewards = torch.Tensor(np.array([reward, reward, reward, reward]))
            ret_inputs = torch.Tensor(np.array([original, rot1, rot2, rot3]))
            return ret_inputs, ret_rewards

    def __len__(self):
        return self.amount_of_training_pts

    def set_probability(self, new_prob):
        self.positive_percent = new_prob


class CustomDatasetFive(Dataset):

    def __init__(self, neg_path_to_folder, pos_path_to_folder):
        self.positive_percent = 0.5

        self.neg_path_to_folder = neg_path_to_folder
        self.all_neg_elements_in_dir_list = os.listdir(neg_path_to_folder)
        self.all_neg_elements_in_dir_list.sort()
        self.neg_size = math.floor(len(self.all_neg_elements_in_dir_list) / 2)

        self.pos_path_to_folder = pos_path_to_folder
        self.all_pos_elements_in_dir_list = os.listdir(pos_path_to_folder)
        self.all_pos_elements_in_dir_list.sort()
        self.pos_size = math.floor(len(self.all_pos_elements_in_dir_list) / 2)

        self.amount_of_training_pts = (self.neg_size + self.pos_size)

    # index isn't actually used -> we want to control how many negative or positive samples we train
    def __getitem__(self, index):
        toggle = random.random()
        # print("** called")
        if toggle < self.positive_percent:
            r = random.randint(0, self.neg_size - 1)
            reward = np.load((self.neg_path_to_folder + "/" + self.all_neg_elements_in_dir_list[2 * r + 2]))
            original = np.load((self.neg_path_to_folder + "/" + self.all_neg_elements_in_dir_list[2 * r + 1]))
            reward = reward_manager.reward_shaper_neg_five(reward, original)
            rot1 = tools.rotate_by_90(original)
            rot2 = tools.rotate_by_90(rot1)
            rot3 = tools.rotate_by_90(rot2)
            # print("debug 1")
            ret_rewards = torch.Tensor(np.array([reward, reward, reward, reward]))
            ret_inputs = torch.Tensor(np.array([original, rot1, rot2, rot3]))
            return ret_inputs, ret_rewards
        else:
            # print("----- called else")
            r = random.randint(0, self.pos_size - 1)
            reward = np.load((self.pos_path_to_folder + "/" + self.all_pos_elements_in_dir_list[2 * r + 2]))
            original = np.load((self.pos_path_to_folder + "/" + self.all_pos_elements_in_dir_list[2 * r + 1]))
            reward = reward_manager.reward_shaper_pos_five(reward, original)
            rot1 = tools.rotate_by_90(original)
            rot2 = tools.rotate_by_90(rot1)
            rot3 = tools.rotate_by_90(rot2)
            # print(reward)
            # dummy1 = np.array([reward, reward, reward, reward])
            ret_rewards = torch.Tensor(np.array([reward, reward, reward, reward]))
            ret_inputs = torch.Tensor(np.array([original, rot1, rot2, rot3]))
            return ret_inputs, ret_rewards

    def __len__(self):
        return self.amount_of_training_pts

    def set_probability(self, new_prob):
        self.positive_percent = new_prob


def encodeAugmentationThree(state):
    return_array = np.zeros((3, 3, 12))
    for i in range (0,3):
        for j in range (0,3):
            return_array[i,j] = encodeElement(state[i,j])
    return return_array


def encodeAugmentationThreeNo90(state):
    return_array = np.zeros((3, 3))
    for i in range (0,3):
        for j in range (0,3):
            if state[i,j] == 90:
                return_array[i,j] = 12
            else:
                return_array[i,j] = state[i,j]
    return return_array


def encodeElement(state_num):
    switcher = {
        -1: np.array([1,0,0,0,0,0,0,0,0,0,0,0]),
        0: np.array([0,1,0,0,0,0,0,0,0,0,0,0]),
        1: np.array([0,0,1,0,0,0,0,0,0,0,0,0]),
        2: np.array([0,0,0,1,0,0,0,0,0,0,0,0]),
        3: np.array([0,0,0,0,1,0,0,0,0,0,0,0]),
        4: np.array([0,0,0,0,0,1,0,0,0,0,0,0]),
        5: np.array([0,0,0,0,0,0,1,0,0,0,0,0]),
        6: np.array([0,0,0,0,0,0,0,1,0,0,0,0]),
        7: np.array([0,0,0,0,0,0,0,0,1,0,0,0]),
        8: np.array([0,0,0,0,0,0,0,0,0,1,0,0]),
        10: np.array([0,0,0,0,0,0,0,0,0,0,1,0]),
        90: np.array([0,0,0,0,0,0,0,0,0,0,0,1]),
    }
    return switcher.get(state_num, np.array([0,0,0,0,0,0,0,0,0,0,0,0]))


class CustomDatasetThreeAug(Dataset):

    def __init__(self, neg_path_to_folder, pos_path_to_folder):
        self.positive_percent = 0.5

        self.neg_path_to_folder = neg_path_to_folder
        self.all_neg_elements_in_dir_list = os.listdir(neg_path_to_folder)
        self.all_neg_elements_in_dir_list.sort()
        self.neg_size = math.floor(len(self.all_neg_elements_in_dir_list) / 2)

        self.pos_path_to_folder = pos_path_to_folder
        self.all_pos_elements_in_dir_list = os.listdir(pos_path_to_folder)
        self.all_pos_elements_in_dir_list.sort()
        self.pos_size = math.floor(len(self.all_pos_elements_in_dir_list) / 2)

        self.amount_of_training_pts = (self.neg_size + self.pos_size)

    # index isn't actually used -> we want to control how many negative or positive samples we train
    def __getitem__(self, index):
        toggle = random.random()
        #print()
        # print("** called")
        if toggle < self.positive_percent:
            r = random.randint(0, self.neg_size - 1)
            reward = np.load((self.neg_path_to_folder + "/" + self.all_neg_elements_in_dir_list[2 * r + 2]))
            original = np.load((self.neg_path_to_folder + "/" + self.all_neg_elements_in_dir_list[2 * r + 1]))
            reward = reward_manager.reward_shaper_neg_three(reward, original)
            rot1 = tools.rotate_by_90(original)
            rot2 = tools.rotate_by_90(rot1)
            rot3 = tools.rotate_by_90(rot2)
            # print("debug 1")
            # print(original)
            ret_rewards = torch.Tensor(np.array([reward, reward, reward, reward]))
            ret_inputs = torch.Tensor(np.array([encodeAugmentationThree(original),
                                                encodeAugmentationThree(rot1),
                                                encodeAugmentationThree(rot2),
                                                encodeAugmentationThree(rot3)]))
            return ret_inputs, ret_rewards
        else:
            # print("----- called else")
            r = random.randint(0, self.pos_size - 1)
            reward = np.load((self.pos_path_to_folder + "/" + self.all_pos_elements_in_dir_list[2 * r + 2]))
            original = np.load((self.pos_path_to_folder + "/" + self.all_pos_elements_in_dir_list[2 * r + 1]))
            reward = reward_manager.reward_shaper_pos_three(reward, original)
            rot1 = tools.rotate_by_90(original)
            rot2 = tools.rotate_by_90(rot1)
            rot3 = tools.rotate_by_90(rot2)
            # print(reward)
            # dummy1 = np.array([reward, reward, reward, reward])
            ret_rewards = torch.Tensor(np.array([reward, reward, reward, reward]))
            # ret_inputs = torch.Tensor(np.array([original, rot1, rot2, rot3]))
            ret_inputs = torch.Tensor(np.array([encodeAugmentationThree(original),
                                                encodeAugmentationThree(rot1),
                                                encodeAugmentationThree(rot2),
                                                encodeAugmentationThree(rot3)]))
            return ret_inputs, ret_rewards

    def __len__(self):
        return self.amount_of_training_pts

    def set_probability(self, new_prob):
        self.positive_percent = new_prob


class CustomDatasetThreeAugNo90(Dataset):

    def __init__(self, neg_path_to_folder, pos_path_to_folder):
        self.positive_percent = 0.5

        self.neg_path_to_folder = neg_path_to_folder
        self.all_neg_elements_in_dir_list = os.listdir(neg_path_to_folder)
        self.all_neg_elements_in_dir_list.sort()
        self.neg_size = math.floor(len(self.all_neg_elements_in_dir_list) / 2)

        self.pos_path_to_folder = pos_path_to_folder
        self.all_pos_elements_in_dir_list = os.listdir(pos_path_to_folder)
        self.all_pos_elements_in_dir_list.sort()
        self.pos_size = math.floor(len(self.all_pos_elements_in_dir_list) / 2)

        self.amount_of_training_pts = (self.neg_size + self.pos_size)

    # index isn't actually used -> we want to control how many negative or positive samples we train
    def __getitem__(self, index):
        toggle = random.random()
        #print()
        # print("** called")
        if toggle < self.positive_percent:
            r = random.randint(0, self.neg_size - 1)
            reward = np.load((self.neg_path_to_folder + "/" + self.all_neg_elements_in_dir_list[2 * r + 2]))
            original = np.load((self.neg_path_to_folder + "/" + self.all_neg_elements_in_dir_list[2 * r + 1]))
            reward = reward_manager.reward_shaper_neg_three(reward, original)
            rot1 = tools.rotate_by_90(original)
            rot2 = tools.rotate_by_90(rot1)
            rot3 = tools.rotate_by_90(rot2)
            # print("debug 1")
            # print(original)
            ret_rewards = torch.Tensor(np.array([reward, reward, reward, reward]))
            ret_inputs = torch.Tensor(np.array([encodeAugmentationThreeNo90(original),
                                                encodeAugmentationThreeNo90(rot1),
                                                encodeAugmentationThreeNo90(rot2),
                                                encodeAugmentationThreeNo90(rot3)]))
            return ret_inputs, ret_rewards
        else:
            # print("----- called else")
            r = random.randint(0, self.pos_size - 1)
            reward = np.load((self.pos_path_to_folder + "/" + self.all_pos_elements_in_dir_list[2 * r + 2]))
            original = np.load((self.pos_path_to_folder + "/" + self.all_pos_elements_in_dir_list[2 * r + 1]))
            reward = reward_manager.reward_shaper_pos_three(reward, original)
            rot1 = tools.rotate_by_90(original)
            rot2 = tools.rotate_by_90(rot1)
            rot3 = tools.rotate_by_90(rot2)
            # print(reward)
            # dummy1 = np.array([reward, reward, reward, reward])
            ret_rewards = torch.Tensor(np.array([reward, reward, reward, reward]))
            ret_inputs = torch.Tensor(np.array([encodeAugmentationThreeNo90(original),
                                                encodeAugmentationThreeNo90(rot1),
                                                encodeAugmentationThreeNo90(rot2),
                                                encodeAugmentationThreeNo90(rot3)]))
            return ret_inputs, ret_rewards

    def __len__(self):
        return self.amount_of_training_pts

    def set_probability(self, new_prob):
        self.positive_percent = new_prob