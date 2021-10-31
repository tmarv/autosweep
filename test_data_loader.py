import torch
from torch.utils.data import DataLoader

import src.tools
from src import custom_data_loader, tools

# how to use batch size?
'''
# this works
number = 4.0
array_n = [4.0, 4.0, 4.0, 4.0]
ttensor = torch.Tensor(array_n)
print("this is number "+str(number))
print("this is array_n "+str(array_n))
print(ttensor)
'''
print("test 1")
pos_location, neg_location = tools.get_save_location_three()
src.tools.get_save_location_three()
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 1}
custom_set = custom_data_loader.CustomDataset(src.tools.get_save_location_three()[0],
                                              src.tools.get_save_location_three()[1])
train_loader = DataLoader(custom_set, **params)
print("test 2")

print("started ")
for i, data in enumerate(train_loader, 0):
    print("hello 1")
    inputs, labels = data

exit()


'''
print("test 3")
exit()
for i in range(0, 2):
    a = custom_set[i]
    print(a)

print("test 3")
# pos_location, neg_location = tools.get_save_test_location_three()
custom_set = custom_data_loader.CustomDataset(tools.get_save_test_location_three()[0],
                                              tools.get_save_test_location_three()[1])
for i in range(0, 2):
    a = custom_set[i]
    print(a)
print("test 4")
custom_set = custom_data_loader.CustomDataset(tools.get_save_test_location_five()[0],
                                              tools.get_save_test_location_five()[1])
for i in range(0, 2):
    a = custom_set[i]
    print(a)
print("test 5")
custom_set = custom_data_loader.CustomDataset(tools.get_save_location_five()[0],
                                              tools.get_save_location_five()[1])
for i in range(0, 2):
    a = custom_set[i]
    print(a)
print("test 6")
'''