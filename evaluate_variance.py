#!/usr/bin/env python3
# Tim Marvel
import math
import os
import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import src.tools
from src import neural_net_lib, reward_manager, tools, custom_data_loader_text
#from src import

#print(tools.get_text_file_names_small()[0])
#print(tools.get_text_file_names_var()[0])
text_file_with_var = tools.get_text_file_names_var()[0]

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = tools.get_device()
neural_net = neural_net_lib.ThreeByThreeSig().to(device)
net_name = os.path.abspath(os.path.join(tools.get_working_dir(), '../saved_nets/neural_net_three_test_variance'))
neural_net.load_state_dict(torch.load(net_name))

# we want them 1 by 1 since we are writting in a datafile
params_three = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}

#true because is small dataset
custom_set_three = custom_data_loader_text.CustomDatasetFromTextFiles3()
#print("this is custom set three: "+str(custom_set_three))
train_loader_three = DataLoader(custom_set_three, **params_three)

_rewards3_text_file_with_var = open(text_file_with_var, 'w')

for i, data in enumerate(train_loader_three):
    inputs, rewards = data
    # make sure it is the same length as batch size
    input_len = len(inputs)
    inputs_res = inputs.reshape([input_len, 3, 3]).to(device)
    rewards = rewards.reshape([input_len, 1]).to(device)
    result = neural_net.forward(inputs_res)
    cluster = 0

    if abs(result - rewards)>0.3:
        cluster = 1
        if result<0 or rewards<0:
            cluster = 2


    inputs_list = inputs.flatten().tolist()
    list = ','.join(str(v) for v in inputs_list)
    #print(str(list))
    _rewards3_text_file_with_var.write(list+","+str(rewards.item())+","+str(result.item())+","+str(cluster)+"\n")
    #print(str(inputs)+str(abs(result-rewards))+"  "+str(result))


_rewards3_text_file_with_var.close()
