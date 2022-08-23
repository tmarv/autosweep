#!/usr/bin/env python3
# Tim Marvel
import math
import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# ml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# custom
import src.tools
from src import reward_manager, tools, neural_net_lib, custom_data_loader_text

def run_five():
    neural_net_five = neural_net_lib.FiveByFiveConv().to(device)
    five_five = np.array([1.0,2.0,3.0,4.0,5.0,5.0,4.0,3.0,2.0,1.0,
                          1.0,2.0,3.0,4.0,5.0,5.0,4.0,3.0,2.0,1.0,
                          1.0,2.0,3.0,4.0,5.0])
    five_five = torch.from_numpy(five_five).to(device)
    five_five = torch.reshape(five_five,(5,5))
    five_five = five_five[None, None, :]
    five_five = five_five.float()
    print(five_five)
    #batch depth w h
    neural_net_five.forward(five_five)

    '''
    net_name_five = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/raw_net_five'))
    neural_net_five.load_state_dict(torch.load(net_name_five, map_location=device))
    neural_net_five.eval()
    '''
device = tools.get_device()
run_five()