#!/usr/bin/env python3
# Tim Marvel
import os
import numpy as np
import pandas as pd
import math
import json
import csv
import matplotlib.pyplot as plt

# TODO: plot histogram

def plot_bar(rewards,label="title"):
    plt.clf()
    rewards = np.sort(rewards)
    plt.plot(np.array(rewards), label=label)
    plt.legend()
    plt.show()

def quantitize_rewards(read_name):
    with open(read_name) as file_obj:
        csv_obj = csv.reader(file_obj)
        raw_feature_points = []
        raw_rewards = []

        for line in csv_obj:
            raw_rewards.append(np.float32(line[9]))
            for i in range(8):
                raw_feature_points.append(np.float32(line[i]))

        min_val_r = np.min(raw_rewards)
        max_val_r = np.max(raw_rewards)
        print("this is min val r: "+str(min_val_r))
        print("this is max valr: "+str(max_val_r))

        min_val_f = np.min(raw_feature_points)
        max_val_f = np.max(raw_feature_points)
        print("this is min val r: "+str(min_val_f))
        print("this is max valr: "+str(max_val_f))
        plot_bar(raw_rewards,"rewards")

# quantitize_rewards("unique_pts.csv")
# quantitize_rewards("unique_normalized_rewards.csv")
# quantitize_rewards("unique_standardized_rewards.csv")
quantitize_rewards("unique_normalized_rewards_m55.csv")