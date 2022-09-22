#!/usr/bin/env python3
# Tim Marvel

import src.neural_net_lib as netlib
import torch
import torch.nn as nn

device = 'cpu'

conv5by5 = netlib.FiveByFiveConv()

grid = [[1.0 , 10.0 , 10.0, 10.0, -1.0],
        [1.0 , 10.0 , 10.0, 10.0, -1.0],
        [1.0 , 10.0 , 10.0, 10.0, -1.0],
        [1.0 , 10.0 , 10.0, 10.0, -1.0],
        [1.0 , 10.0 , 10.0, 10.0, -1.0]]
# grid goes to torch tensor
grid_tensor = torch.as_tensor(grid)
print(grid_tensor.size())
# check what dimensions is the batch size?
grid_tensor = grid_tensor[None, None, :]
result = conv5by5.forward(grid_tensor)
print("finished: result")
print(result)