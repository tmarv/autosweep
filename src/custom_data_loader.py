#!/usr/bin/env python3
# Tim Marvel
import os
from random import random

import numpy as np
from torch.utils.data import Dataset
import src.tools
from src import reward_manager, tools


class CustomDataset(Dataset):

    def __init__(self, neg_path_to_folder, pos_path_to_folder):
        self.positive_percent = 0.6

        self.all_neg_elements_in_dir_list = os.listdir(neg_path_to_folder)
        self.all_neg_elements_in_dir_list.sort()
        self.neg_size = len(self.all_neg_elements_in_dir_list)/2

        self.all_pos_elements_in_dir_list = os.listdir(pos_path_to_folder)
        self.all_pos_elements_in_dir_list.sort()
        self.pos_size = len(self.all_pos_elements_in_dir_list)/2

        self.amount_of_training_pts = (self.neg_size + self.pos_size)

    # index isn't actually used -> we want to control how many negative or positive samples we train
    def __getitem__(self, index):
        toggle = random.random()
        if toggle < self.positive_percent:
            r = random.randint(0, self.neg_size - 1)
            reward = np.load((self.all_neg_elements_in_dir_list + "/" + self.all_neg_elements_in_dir_list[2 * r + 2]))
            original = np.load((self.all_pos_elements_in_dir_list + "/" + self.all_pos_elements_in_dir_list[2 * r + 1]))
            reward = reward_manager.reward_shaper_neg_three(reward, original)
            rot1 = tools.rotate_by_90(original)
            rot2 = tools.rotate_by_90(rot1)
            rot3 = tools.rotate_by_90(rot2)
            return 0.1
        else:
            r = random.randint(0, self.pos_size - 1)
            reward = np.load((self.all_pos_elements_in_dir_list + "/" + self.all_pos_elements_in_dir_list[2 * r + 2]))
            original = np.load((self.all_pos_elements_in_dir_list + "/" + self.all_pos_elements_in_dir_list[2 * r + 1]))
            reward = reward_manager.reward_shaper_pos_three(reward, original)
            rot1 = tools.rotate_by_90(original)
            rot2 = tools.rotate_by_90(rot1)
            rot3 = tools.rotate_by_90(rot2)
            return 0.1

    def __len__(self):
        return self.amount_of_training_pts

    def set_probability(self, new_prob):
        self.positive_percent = new_prob
