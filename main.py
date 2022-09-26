import train_networks
import estimate_variance
import torch

'''
train raw network
'''
torch.set_num_threads(3)
#train_networks.train_three_by_three_raw_net(epoch=10000, plot_result=False, backup_name="raw_net_three")

'''
add clusters based on diff to raw net
'''
torch.set_num_threads(3)
#estimate_variance.add_variance_and_cluster_three(thresh=0.2)

'''
train a clustering network
'''

#train_networks.train_cluster_net_three(epoch=5000, plot_result=False, backup_name="cluster_net_three", batch_size=4096)
#train_networks.train_cluster_net_three(epoch=6000, plot_result=False, backup_name="cluster_net_three", batch_size=16384)
#train_networks.train_cluster_net_three(epoch=5000, plot_result=True, backup_name="cluster_net_three")

'''
# train the cluster specific nets
'''
def train_cluster_specific_nets():
    torch.set_num_threads(3)
    train_networks.train_three_by_three_for_one_cluster(0, epoch=8000, plot_result=False, backup_name="net_three", batch_size=16384)
    train_networks.train_three_by_three_for_one_cluster(1, epoch=6000, plot_result=False, backup_name="net_three", batch_size=16384)
    train_networks.train_three_by_three_for_one_cluster(2, epoch=6000, plot_result=False, backup_name="net_three", batch_size=16384)

#train_cluster_specific_nets()

# five by five network
#torch.set_num_threads(4)
#train_networks.train_five_by_five_raw_net(epoch=40000, plot_result=True, backup_name="raw_net_five", learning_rate=0.0003, batch_size=16384)
#train_networks.train_five_by_five_conv(epoch=200, plot_result=True, backup_name="raw_net_five_conv", learning_rate=0.0003, batch_size=16384)
#estimate_variance.add_variance_and_cluster_five_conv(backup_name="raw_net_five_conv")
train_networks.train_cluster_net_five_conv(epoch=1)