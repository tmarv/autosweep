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
        self.positive_percent = 0.0

        self.neg_path_to_folder = neg_path_to_folder
        self.all_neg_elements_in_dir_list = os.listdir(neg_path_to_folder)
        self.all_neg_elements_in_dir_list.sort()
        self.neg_size = math.floor(len(self.all_neg_elements_in_dir_list)/2)

        self.pos_path_to_folder = pos_path_to_folder
        self.all_pos_elements_in_dir_list = os.listdir(pos_path_to_folder)
        self.all_pos_elements_in_dir_list.sort()
        self.pos_size = math.floor(len(self.all_pos_elements_in_dir_list)/2)

        self.amount_of_training_pts = (self.neg_size + self.pos_size)

    # index isn't actually used -> we want to control how many negative or positive samples we train
    def __getitem__(self, index):
        toggle = random.random()
        print("** called")
        if toggle < self.positive_percent:
            r = random.randint(0, self.neg_size - 1)
            reward = np.load((self.neg_path_to_folder + "/" + self.all_neg_elements_in_dir_list[2 * r + 2]))
            original = np.load((self.neg_path_to_folder + "/" + self.all_neg_elements_in_dir_list[2 * r + 1]))
            reward = reward_manager.reward_shaper_neg_three(reward, original)
            rot1 = tools.rotate_by_90(original)
            rot2 = tools.rotate_by_90(rot1)
            rot3 = tools.rotate_by_90(rot2)
            print("debug 1")
            ret_1 = torch.Tensor(reward)
            ret_2 = torch.Tensor([original, rot1, rot2, rot3])
            return ret_1, ret_2
        else:
            print("----- called else")
            r = random.randint(0, self.pos_size - 1)
            reward = np.load((self.pos_path_to_folder + "/" + self.all_pos_elements_in_dir_list[2 * r + 2]))
            original = np.load((self.pos_path_to_folder + "/" + self.all_pos_elements_in_dir_list[2 * r + 1]))
            reward = reward_manager.reward_shaper_pos_three(reward, original)
            rot1 = tools.rotate_by_90(original)
            rot2 = tools.rotate_by_90(rot1)
            rot3 = tools.rotate_by_90(rot2)
            print("debug 2")
            print(reward)
            print("debug 3")
            dummy1 = np.array([reward, reward, reward, reward])
            print(dummy1)
            ret_1 = torch.Tensor(dummy1)
            print(ret_1)
            ret_2 = torch.Tensor([original, rot1, rot2, rot3])
            print("---")
            print(ret_2)
            exit()
            return ret_1, ret_2

    def __len__(self):
        return self.amount_of_training_pts

    def set_probability(self, new_prob):
        self.positive_percent = new_prob
