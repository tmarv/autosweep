import train_networks
import estimate_variance


'''
train raw network
'''
#train_networks.train_three_by_three_raw_net(epoch=10000, plot_result=False, backup_name="raw_net_three")

'''
add clusters based on diff to raw net
'''
estimate_variance.add_variance_and_cluster(thresh=0.2)

'''
train a clustering network
'''
#train_networks.train_cluster_net(epoch=5000, plot_result=False, backup_name="cluster_net_three")
train_networks.train_cluster_net(epoch=5000, plot_result=True, backup_name="cluster_net_three")

'''
# train the cluster specific nets
'''
def train_cluster_specific_nets():
    train_networks.train_three_by_three_for_one_cluster(0, epoch=20000, plot_result=False, backup_name="net_three")
    train_networks.train_three_by_three_for_one_cluster(1, epoch=20000, plot_result=False, backup_name="net_three")
    train_networks.train_three_by_three_for_one_cluster(2, epoch=20000, plot_result=False, backup_name="net_three")

#train_cluster_specific_nets()