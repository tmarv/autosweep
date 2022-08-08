import numpy as np
import os
import pandas as pd
from src import tools
import matplotlib.pyplot as plt


def plotThreeByThree():
    text_file_name = tools.get_text_file_names_var()[0]
    rewards = np.array(pd.read_csv(text_file_name, header=None, usecols=[9]))
    network_rewards = np.array(pd.read_csv(text_file_name, header=None, usecols=[10]))
    diff = np.subtract(network_rewards,rewards)
    #plt.plot(rewards)
    #plt.plot(network_rewards)
    plt.plot(diff)
    plt.show()


def modifyValues(threshold=0.3):
    text_var_file_name = tools.get_text_file_names_var()[0]
    input_data = np.array(pd.read_csv(text_var_file_name, header=None, usecols=[0,1,2,3,4,5,6,7,8,10]))
    rewards = np.array(pd.read_csv(text_var_file_name, header=None, usecols=[9]))
    network_rewards = np.array(pd.read_csv(text_var_file_name, header=None, usecols=[10]))
    diff = np.subtract(network_rewards, rewards)
    text_file_with_var = tools.get_text_file_names_clean()[0]
    _rewards3_text_file_clean = open(text_file_with_var, 'w')

    for i in range (len(diff)):
        if(abs(diff[i]) < 0.3):
            #input_data[i][9]*=5
            inputs_list = input_data[i].flatten().tolist()
            list = ','.join(str(v) for v in inputs_list)
            _rewards3_text_file_clean.write(list+"\n")
        else:
            if(input_data[i][9]>0):
                input_data[i][9]/=5
            else:
                input_data[i][9]-=10
            inputs_list = input_data[i].flatten().tolist()
            list = ','.join(str(v) for v in inputs_list)
            _rewards3_text_file_clean.write(list + "\n")
    _rewards3_text_file_clean.close()

#plotThreeByThree()
modifyValues(0.3)