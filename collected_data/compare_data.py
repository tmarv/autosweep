#!/usr/bin/env python3
# Tim Marvel
import os
import numpy as np
import pandas as pd
# import torch
# import random
import math
import json


def open_and_sort()->pd.DataFrame:
    df = pd.read_csv("rewards3.txt")
    df.columns.values[0] = 'd1'
    df.columns.values[1] = 'd2'
    df.columns.values[2] = 'd3'
    df.columns.values[3] = 'd4'
    df.columns.values[4] = 'd5'
    df.columns.values[5] = 'd6'
    df.columns.values[6] = 'd7'
    df.columns.values[7] = 'd8'
    df.columns.values[8] = 'd9'
    df.columns.values[9] = 'reward'
    df = df.sort_values(by=['d1','d2','d3','d4','d5','d6','d7','d8','d9'], ascending=True)
    return df


def perform_comparison(frame_a, frame_b):
    same_result = frame_a.reward == frame_b.reward
    same_data = frame_a.d1 == frame_b.d1
    same_data = same_data and frame_a.d2 == frame_b.d2
    same_data = same_data and frame_a.d3 == frame_b.d3
    same_data = same_data and frame_a.d4 == frame_b.d4
    same_data = same_data and frame_a.d5 == frame_b.d5
    same_data = same_data and frame_a.d6 == frame_b.d6
    same_data = same_data and frame_a.d7 == frame_b.d7
    same_data = same_data and frame_a.d8 == frame_b.d8
    same_data = same_data and frame_a.d9 == frame_b.d9
    return same_result, same_data


def dump_to_file(dataList, filename):
    csv_file = open(filename,'w')
    for pdframe in dataList:
        #df = pdframe.to_csv(header=None, index=False).strip('\n').split('\n')
        #df = pdframe.to_csv(header=None, index=False).strip('\n').split('\n')
        df = pdframe.transpose()
        #print(df)
        #print(pdframe)
        #exit()
        data_line = ''
        for val in df:
            data_line+=str(val)+","
            #csv_file.write(str(val)+",")
        csv_file.write(data_line[:-1])
        csv_file.write("\n")
        #df.to_csv(filename)
        #print(df)
        #exit()
    csv_file.close()
    #pd.concat(dataList).to_csv(filename, index=False,sep=',')


def count_duplicates():
    print("started count_duplicates for 3 by 3")
    data = open_and_sort()
    print(data)
    if data.empty:
        print("WARNING: the data frame read from the text file is empty")
        return


    #print(hash(data.iloc[0].items))
    #print(hash(data.iloc[1].items))
    #exit()
    total_len = data.shape[0]

    unique_datapts = []
    current_row = data.iloc[0]

    multiple_results = []
    known_rewards = []
    known_rewards.append(current_row.reward)
    unique_datapts.append(current_row)
    cntr = 1
    all_same_cntr = 1
    total_reward = 0
    reward_cntr = 1
    # counts the largest amount of different results for same data in the dataset
    max_cntr = 1
    previous_max = []
    # holds the data for the most common
    most_common_data_diff_result = 0
    for i in range (0, total_len-1):
        same_result, same_data = perform_comparison(data.iloc[i], data.iloc[i+1])
        if same_result and same_data:
            total_reward += data.iloc[i].reward
            reward_cntr += 1
            continue
        elif same_data:
            reward_lcl = data.iloc[i].reward
            total_reward += reward_lcl
            reward_cntr += 1
            if reward_lcl in known_rewards:
                continue
            multiple_results.append(data.iloc[i])
            known_rewards.append(reward_lcl)
        else:
            if max_cntr < reward_cntr:
                most_common_data_diff_result = current_row
                max_cntr = reward_cntr
                previous_max.append(max_cntr)
            if reward_cntr > 1:
                avg_reward = total_reward/reward_cntr
                data.iloc[i].reward = avg_reward
                unique_datapts.append(data.iloc[i])
                known_rewards = []
                known_rewards.append(data.iloc[i+1].reward)
                total_reward = 0
                reward_cntr = 1
            elif reward_cntr == 1:
                unique_datapts.append(data.iloc[i])
                known_rewards = []
                known_rewards.append(data.iloc[i+1].reward)
                total_reward = 0
                reward_cntr = 1

    '''
    for i in range (1, total_len):
        same_result, same_data = perform_comparison(data.iloc[i-1], data.iloc[i])
        if same_result and same_data:

            continue
            # TODO need a better data structure (dict)
        elif same_data:
                reward_lcl = data.iloc[i].reward
                # print(data.iloc[i].items)
                if reward_lcl in known_rewards:
                    continue
                #print("same but diff "+str(data.iloc[i].reward)+"  "+str(current_row.reward))
                cntr += 1
                multiple_results.append(current_row)
                known_rewards.append(reward_lcl)
                #unique_datapts.append(data.iloc[i])
                current_row = data.iloc[i]
        elif (i+1<total_len):
                if max_cntr < cntr:
                    most_common_data_diff_result = current_row
                    max_cntr = cntr
                    previous_max.append(max_cntr)
                if cntr == 1:
                    unique_datapts.append(data.iloc[i])
                #else:
                #    print("single")
                current_row = data.iloc[i]
                known_rewards = []
                known_rewards.append(current_row.reward)
                #unique_datapts.append(current_row)
                cntr = 1
    '''
    print(len(unique_datapts))
    print("this is max cntr: "+str(max_cntr))
    print("this is multiple_results: "+str(len(multiple_results)))
    print("most_common_data_diff_result:"+str(most_common_data_diff_result))
    print(previous_max)
    #print(unique_datapts)
    dump_to_file(unique_datapts, "unique_pts.csv")
    dump_to_file(multiple_results, "repeats_pts.csv")
    '''
    should_be_true1, should_be_true2 = perform_comparison(data.iloc[0], data.iloc[1])
    should_not_true1, should_not_true2 = perform_comparison(data.iloc[0], data.iloc[200])

    print("true: "+str(should_be_true1)+"  "+str(should_be_true2))
    print("false: "+str(should_not_true1)+"  "+str(should_not_true2))
    exit()
    current_row = data.iloc[1]
    if (data.iloc[1].d1 == data.iloc[0].d1):
        print("is the same")
    else:
        print("there is a problem")
        print(data.iloc[1].items)
        print(data.iloc[0].items)
    # print(data.keys())
    '''
print("starting data comparison")
count_duplicates()
