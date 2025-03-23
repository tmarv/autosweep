import csv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from src import model_zoo
from src import tools

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# DRY violation
class CustomDatasetFromCSV(Dataset):
    def __init__(self, path_to_file):
        self.grid_values = []
        self.rewards = []
        with open(path_to_file) as file_obj:
            csv_obj = csv.reader(file_obj)
            counter = 0
            for line in csv_obj:
                self.rewards.append(np.float32(line[49]))
                self.grid_values.append([line[0:7], line[7:14], line[14:21], line[21:28], line[28:35], line[35:42], line[42:49]])
        self.len = len(self.rewards)
        self.grid_values = torch.Tensor(np.array(np.float32(self.grid_values))).to(device)
        self.rewards = torch.Tensor(np.array(np.float32(self.rewards))).to(device)

    def __getitem__(self, item):
        return self.grid_values[item], self.rewards[item]

    def __len__(self):
        return self.len

def runInference(input, neural_net):
    local_tensor = input.unsqueeze(1)
    neural_net_pred = neural_net.forward(local_tensor)
    return neural_net_pred.detach().cpu()


def run_eval(net_name:str, sz:int, batch_size:int):
    main_net = model_zoo.SevenBySeven2ConvLayerXLeakyReLUSigmoidEnd(sz, 0.0).to(device)
    main_net_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/{}'.format(net_name)))
    main_net.load_state_dict(torch.load(main_net_name, map_location=device))
    main_net.eval()
    print("Loading the dataset from the CSV file")
    # load the test data
    dataset = CustomDatasetFromCSV('collected_data/unique_normalized_7_rewards_m25.csv')
    # load it into a pytorch dataset
    print("Loading the dataset into a dataloader")
    data_loader_seven = DataLoader(dataset, batch_size = batch_size, shuffle=False)
    # call inference on the dataset
    print("Starting inference")
    resluts_array = []
    ground_truth_array = []
    for batch_idx, (data, targets) in enumerate(data_loader_seven):
        prediction = runInference(data, main_net)
        ground_truth_array.extend(targets.detach().cpu())
        resluts_array.extend(prediction)
    print("Finished inference")
    # get the difference between ground truth and prediction
    plotting_diff = [a - b for a, b in zip(resluts_array, ground_truth_array)]
    d = len(plotting_diff)
    # Create a bar plot -> because of the large number of datapoints we have to manually downsize the elements
    # store the results in an counter array
    borders = [-.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
    counters = [0] * len(borders)
    print("Preparing the histogram")
    for diff in plotting_diff:
        for i in range(len(borders)):
            if diff < borders[i]:
                counters[i]+=1
                break
    print(counters)
    x_positions = [b + i * (0.1 + .15) for i, b in enumerate(borders)]
    counters = [x / d for x in counters]
    print(counters)
    print("Plotting the histogram")
    plt.bar(borders, counters, width=0.1, edgecolor='r')
    plt.title('Histogram plot')
    plt.xlabel('Data')
    plt.ylabel('Frequency (normalizerd)')
    plt.xlim([borders[0]-0.1, borders[-1]+0.1])
    plt.savefig(net_name+'.png')
    #plt.show()



device = tools.get_device()
print(f"running on {device}")
# we can use a huge batch size, as long as it fits onto the GPU. Only for evalutation purporses
run_eval('seven_conv_16_drop_0_bs_64_m25_nd_l1', 16, 256*2048)
