#!/usr/bin/env python3
# Tim Marvel

import train_networks
import estimate_variance
# global imports
import json
import torch

config_file = open("config/config_retrain_conv_5.json")
config = json.load(config_file)
net_dim = config["dimensions"]
config_raw = config["train_raw_net"]
config_cluster = config["train_clustering_net"]
config_individual_cluster = config["train_individual_cluster_net"]
config_file.close()

if net_dim == 3:
    print("training a three by three network")
    if not config_raw[0]["skip"]:
        print("training raw three by three net")
        # train_networks.train_three_by_three_raw_net(epoch=10000, plot_result=False, backup_name="raw_net_three")

    if not config_cluster[0]["skip"]:
        if not config_cluster[0]["skip_variance_eval"]:
            print("evaluating variance")
            # estimate_variance.add_variance_and_cluster_three(thresh=0.2)
        print("training clustering net")
        # train_networks.train_cluster_net_three(epoch=5000, plot_result=False, backup_name="cluster_net_three", batch_size=4096)
    if not config_individual_cluster[0]["skip"]:
        train_networks.train_three_by_three_for_one_cluster(0,
                                                            epoch=8000,
                                                            plot_result=False,
                                                            backup_name="net_three",
                                                            batch_size=16384)
        train_networks.train_three_by_three_for_one_cluster(1,
                                                            epoch=6000,
                                                            plot_result=False,
                                                            backup_name="net_three",
                                                            batch_size=16384)
        train_networks.train_three_by_three_for_one_cluster(2,
                                                            epoch=6000,
                                                            plot_result=False,
                                                            backup_name="net_three",
                                                            batch_size=16384)

elif net_dim == 5:
    print("training a five by five net")
    if not config_raw[0]["skip"]:
        train_networks.train_five_by_five_conv(plot_result=True,
                                               epoch=config_raw[0]["epochs"],
                                               backup_name=config_raw[0]["net_backup_name"],
                                               learning_rate=config_raw[0]["learning_rate"],
                                               batch_size=config_raw[0]["batch_size"],
                                               use_pretrained=config_raw[0]["use_pretrained_net"],
                                               pretrained_name=config_raw[0]["pretrained_net_name"],
                                               training_loss_graph=config_raw[0]["training_loss_graph"])
    if not config_cluster[0]["skip"]:
        if not config_cluster[0]["skip_variance_eval"]:
            estimate_variance.add_variance_and_cluster_five_conv(config_cluster[0]["raw_evaluation_net"],
                                                                 plot_result=True,
                                                                 thresh=config_cluster[0]["variance_threshold"],
                                                                 backup_plot_name = config_cluster[0]["variance_eval_graph_name"])

        train_networks.train_cluster_net_five_conv(epoch=config_cluster[0]["epochs"],
                                                   batch_size=config_cluster[0]["batch_size"],
                                                   plot_result=config_cluster[0]["plot_results"],
                                                   backup_name=config_cluster[0]["net_backup_name"],
                                                   learning_rate=config_cluster[0]["learning_rate"],
                                                   use_pretrained=config_cluster[0]["use_pretrained_net"],
                                                   pretrained_net_name=config_cluster[0]["pretrained_net_name"],
                                                   training_loss_graph=config_cluster[0]["training_loss_graph"])

    if not config_individual_cluster[0]["skip"]:
        train_networks.train_five_by_five_for_one_cluster(0,
                                                          epoch=config_individual_cluster[0]["epochs_0"],
                                                          learning_rate=config_individual_cluster[0]["learning_rate"],
                                                          plot_result=config_individual_cluster[0]["plot_results"],
                                                          backup_name=config_individual_cluster[0]["net_backup_name_0"],
                                                          training_loss_graph=config_individual_cluster[0]["training_loss_graph_cluster_0"],
                                                          batch_size=config_individual_cluster[0]["batch_size"],
                                                          use_pretrained=config_individual_cluster[0]["use_pretrained_net"],
                                                          pretrained_name=config_individual_cluster[0]["pretrained_net_name"]
                                                          )
        train_networks.train_five_by_five_for_one_cluster(1,
                                                          epoch=config_individual_cluster[0]["epochs_1"],
                                                          learning_rate=config_individual_cluster[0]["learning_rate"],
                                                          plot_result=config_individual_cluster[0]["plot_results"],
                                                          backup_name=config_individual_cluster[0]["net_backup_name_1"],
                                                          training_loss_graph=config_individual_cluster[0]["training_loss_graph_cluster_1"],
                                                          batch_size=config_individual_cluster[0]["batch_size"],
                                                          use_pretrained=config_individual_cluster[0]["use_pretrained_net"],
                                                          pretrained_name=config_individual_cluster[0]["pretrained_net_name"])
        train_networks.train_five_by_five_for_one_cluster(2,
                                                          epoch=config_individual_cluster[0]["epochs_2"],
                                                          learning_rate=config_individual_cluster[0]["learning_rate"],
                                                          plot_result=config_individual_cluster[0]["plot_results"],
                                                          backup_name=config_individual_cluster[0]["net_backup_name_2"],
                                                          training_loss_graph=config_individual_cluster[0]["training_loss_graph_cluster_2"],
                                                          batch_size=config_individual_cluster[0]["batch_size"],
                                                          use_pretrained=config_individual_cluster[0]["use_pretrained_net"],
                                                          pretrained_name=config_individual_cluster[0]["pretrained_net_name"])
else:
    print("no known training dimensions mentioned in config: will exit")