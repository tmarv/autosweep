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

def quantitize_rewards3by3(read_name):
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


def quantitize_rewards5by5(read_name):
    with open(read_name) as file_obj:
        csv_obj = csv.reader(file_obj)
        raw_feature_points = []
        raw_rewards = []

        for line in csv_obj:
            raw_rewards.append(np.float32(line[25]))
            for i in range(24):
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


def quantitize_rewards7by7(read_name):
    with open(read_name) as file_obj:
        csv_obj = csv.reader(file_obj)
        raw_feature_points = []
        raw_rewards = []

        for line in csv_obj:
            raw_rewards.append(np.float32(line[49]))
            for i in range(48):
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


quantitize_rewards5by5("unique_normalized_7_rewards_m25.csv")
#quantitize_rewards5by5("unique_normalized_5_rewards_m25.csv")
#quantitize_rewards3by3("unique_normalized_3_rewards_m25.csv")