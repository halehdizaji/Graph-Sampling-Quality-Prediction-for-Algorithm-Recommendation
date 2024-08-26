################################### In this file my other implemented sampling algorithms are also used ####################################
################################### In addition in this file generated graphs of different sizes and params are utilized. #################
################################## In this file also graphs_info.pkl file is used for reading
################################## In this file also real graph data can be used.
################################## In this file also mean and variance of sampling algorithms results are calculated.
#import Graph_Sampling
import random
import pickle
import json as js
import networkx as nx
import networkit as nk
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import configparser
import argparse
import sys
sys.path.append('../')

from graph_processing.Calc_Distributions import calc_weak_connected_components_sizes_dist, calc_shortest_paths_CC_list_nx, calc_hop_plot_CC_dist_nx, calc_hop_plot_dist_nx, calc_normalized_cdf_continuous, calc_normalized_degree_dist, calc_KS_D_statistic, approx_sp_stat_distr_nk

# samplings

import networkit as nk
from typing import Union
import sys
from littleballoffur.sampler import Sampler

NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph

from typing import Union
from littleballoffur.sampler import Sampler
NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph

import random
import math
import networkx as nx
import numpy as np

from typing import Union
from littleballoffur.sampler import Sampler

NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph


def load_graph(graph_folder, graph_file_name):
    print('graph name:', graph_file_name)
    graph_ID = graph_file_name[3:-7] #indexing separates ID_ from start of graph name and .pickle from the end of its name.
    graph = pickle.load(
                open(graph_folder + graph_file_name, 'rb'))
    #print('graph nodes ', graph.nodes())
    input_graph = (graph_ID, graph)

    return input_graph


def load_sample_graphs(graphs_folder, input_graphs = {}):
    for entry in os.listdir(graphs_folder):
        if os.path.isfile(os.path.join(graphs_folder, entry)):
            sampling_alg = entry[0:-13] #indexing separates ID_ from start of graph name and .pickle from the end of its name.
            print(sampling_alg)
            sampling_rate = entry[-12:-9]
            print(sampling_rate)
            if not(sampling_alg in input_graphs):
                input_graphs[sampling_alg] = {}
            if not (sampling_rate in input_graphs[sampling_alg]):
                input_graphs[sampling_alg][sampling_rate] = []
            if os.stat(graphs_folder + entry).st_size == 0:
                graph = nx.Graph()
            else:
                graph = pickle.load(
                    open(graphs_folder + entry, 'rb'))
            input_graphs[sampling_alg][sampling_rate].append(graph)

    return input_graphs


def evaluate_samplings(input_graph, sample_graphs ):
    if os.path.exists(sampling_experiment_folder + 'sampling_results'):
        try:
            graphs_sampling_results = pickle.load(open(sampling_experiment_folder + 'sampling_results', 'rb'))
        except:
            graphs_sampling_results = js.load(open(sampling_experiment_folder + 'sampling_results', 'r'))
    else:
        graphs_sampling_results = {}
    graph_ID = input_graph[0]
    print('graph ', graph_ID)
    graph = input_graph[1]
    ########################## calculating distributions and required features ########
    print('calculating degree dist')
    node_nums = len(graph.nodes())
    orig_graph_degree_dist = calc_normalized_degree_dist(graph, node_nums)
    print('calculating clust coeff distr')
    clust_coeff = nx.clustering(graph)
    orig_graph_clust_coef_dist = calc_normalized_cdf_continuous(clust_coeff, cc_interval = cc_interval)
    js.dump(orig_graph_clust_coef_dist, open( root_folder + '/features/' + graph_file_name  + '_CC_distr_interval_' + str(cc_interval)  + '.json', 'w'))
    # check for existence of saved hop plot distr
    #
    #print('calculating wcc sizes distr')
    print('hop plots')
    # check for availability of hop plots distr of orig graph
    # hop plots all
    if not (os.path.exists(root_folder + '/features/' + graph_file_name + '/' + graph_file_name + '_sp.json') or os.path.exists(root_folder + '/features/' + graph_file_name + '/' + graph_file_name + '_sp_stat_distr_approx_nk.json')) or not (os.path.exists(root_folder + '/features/' + graph_file_name + '/' + graph_file_name + '_sp_lcc.json') or os.path.exists(root_folder + '/features/' + graph_file_name + '/' + graph_file_name + '_sp_lcc_stat_distr_approx_nk.json')):
        # convert graph to nk graph
        # todo
        graph_nk = nk.nxadapter.nx2nk(graph)
        orig_graph_sp_stat_distr, orig_graph_sp_stat_distr_lcc = approx_sp_stat_distr_nk(graph_nk)
        js.dump(orig_graph_sp_stat_distr, open(root_folder + '/features/' + graph_file_name  + '_sp_stat_distr_approx_nk.json', 'w'))
        js.dump(orig_graph_sp_stat_distr_lcc, open(root_folder + '/features/' + graph_file_name  + '_sp_lcc_stat_distr_approx_nk.json', 'w'))
    if os.path.exists(root_folder + '/features/' + graph_file_name + '/' + graph_file_name + '_sp.json'):
        orig_graph_sp_stat_distr = js.load(open(root_folder + '/features/' + graph_file_name + '/' + graph_file_name + '_sp.json', 'r'))
    elif os.path.exists(root_folder + '/features/' + graph_file_name + '/' + graph_file_name + '_sp_stat_distr_approx_nk.json'):
        orig_graph_sp_stat_distr = js.load(open(root_folder + '/features/' + graph_file_name + '/' + graph_file_name + '_sp_stat_distr_approx_nk.json', 'r'))
    if os.path.exists(root_folder + '/features/' + graph_file_name + '/' + graph_file_name + '_sp_lcc.json'):
        orig_graph_sp_stat_distr_lcc = js.load(open(root_folder + '/features/' + graph_file_name + '/' + graph_file_name + '_sp_lcc.json', 'r'))
    elif os.path.exists(root_folder + '/features/' + graph_file_name + '/' + graph_file_name + '_sp_lcc_stat_distr_approx_nk.json'):
        orig_graph_sp_stat_distr_lcc = js.load(open(root_folder + '/features/' + graph_file_name + '/' + graph_file_name + '_sp_lcc_stat_distr_approx_nk.json', 'r'))
    
    orig_graph_hop_plot_dist = orig_graph_sp_stat_distr['shortest paths lengths distr']
    orig_graph_hop_plot_LCC_dist = orig_graph_sp_stat_distr_lcc['shortest paths lengths distr']
    ################################## Sampling ###############################
    for sampling_algorithm in sample_graphs:
        samples_info = sample_graphs[sampling_algorithm]
        for sampling_percent in samples_info:
            print('sampling rate ', sampling_percent)
            Trial_ID = graph_ID + '_' + sampling_algorithm + '_' + str(sampling_percent)
            #graphs_sampling_results[Trial_ID] = {}
            graphs_sampling_results[Trial_ID]['graph_ID'] = graph_ID
            graphs_sampling_results[Trial_ID]['sampling_percent'] = float(sampling_percent)
            graphs_sampling_results[Trial_ID]['sampling_algorithm'] = sampling_algorithm
            KS_D_dists = []
            KS_CC_dists = []
            KS_hop_plots_dists = []
            KS_hop_plots_LCC_dists = []
            samples = samples_info[sampling_percent]
            for sample_graph in samples:
                sample_node_nums = len(sample_graph.nodes())
                print('calculating normal degree dist cdf')
                print('node num', sample_node_nums)
                if sample_node_nums == 0:
                    KS_D_dist = 1
                    print('calculating KS clust coeff distr ')
                    KS_CC_dist = 1
                    print('calculating distances finished')
                    KS_D_dists.append(KS_D_dist)
                    KS_CC_dists.append(KS_CC_dist)
                    KS_hop_plots_dists.append(1)
                    KS_hop_plots_LCC_dists.append(1)
                else:
                    sample_graph_deg_dist = calc_normalized_degree_dist(sample_graph, sample_node_nums)
                    print('calculating normal clust coeff dist cdf')
                    clust_coeff = nx.clustering(sample_graph)
                    sample_graph_clust_coef_dist = calc_normalized_cdf_continuous(clust_coeff, cc_interval = cc_interval)
                    print('hop plots')
                    #convert sample graph to nk graph
                    sample_graph_nk = nk.nxadapter.nx2nk(sample_graph)
                    sample_graph_sp_stat_distr, sample_graph_sp_stat_distr_lcc = approx_sp_stat_distr_nk(sample_graph_nk)
                    sample_graph_hop_plot_dist = sample_graph_sp_stat_distr['shortest paths lengths distr']
                    sample_graph_hop_plot_LCC_dist = sample_graph_sp_stat_distr_lcc['shortest paths lengths distr']
                    ####################### calculating divergences #####################
                    print('calculating KS deg distr ')
                    KS_D_dist = calc_KS_D_statistic(orig_graph_degree_dist, sample_graph_deg_dist)
                    print('calculating KS clust coeff distr ')
                    KS_CC_dist = calc_KS_D_statistic(orig_graph_clust_coef_dist, sample_graph_clust_coef_dist)
                    print('calculating KS hop plots distr ')
                    KS_hop_plots_dist = calc_KS_D_statistic(orig_graph_hop_plot_dist, sample_graph_hop_plot_dist)
                    print('calculating KS hop plots LCC distr ')
                    KS_hop_plots_LCC_dist = calc_KS_D_statistic(orig_graph_hop_plot_LCC_dist, sample_graph_hop_plot_LCC_dist)
                    print('calculating distances finished')
                    KS_D_dists.append(KS_D_dist.item())
                    KS_CC_dists.append(KS_CC_dist.item())
                    KS_hop_plots_dists.append(KS_hop_plots_dist)
                    KS_hop_plots_LCC_dists.append(KS_hop_plots_LCC_dist)
            KS_D_dists = np.array(KS_D_dists)
            KS_CC_dists = np.array(KS_CC_dists)
            KS_hop_plots_dists =  np.array(KS_hop_plots_dists)
            KS_hop_plots_LCC_dists = np.array(KS_hop_plots_LCC_dists)
            avg_KS_D_dist = KS_D_dists.mean()
            avg_KS_CC_dist = KS_CC_dists.mean()
            avg_KS_hop_plots_dist = KS_hop_plots_dists.mean()
            avg_KS_hop_plots_LCC_dist = KS_hop_plots_LCC_dists.mean()

            KS_D_dist_var = KS_D_dists.var()
            KS_CC_dist_var = KS_CC_dists.var()
            KS_hop_plots_dist_var = KS_hop_plots_dists.var()
            KS_hop_plots_LCC_dist_var = KS_hop_plots_LCC_dists.var()

            graphs_sampling_results[Trial_ID]['KS Degree Distr'] = avg_KS_D_dist
            graphs_sampling_results[Trial_ID]['KS Clustering Coefficient Distr'] = avg_KS_CC_dist
            graphs_sampling_results[Trial_ID]['KS hop plots Distr'] = avg_KS_hop_plots_dist
            graphs_sampling_results[Trial_ID]['KS hop plots LCC Distr'] = avg_KS_hop_plots_LCC_dist
            graphs_sampling_results[Trial_ID]['KS Degree Distr var'] = KS_D_dist_var
            graphs_sampling_results[Trial_ID]['KS Clustering Coefficient Distr var'] = KS_CC_dist_var
            graphs_sampling_results[Trial_ID]['KS hop plots Distr var'] = KS_hop_plots_dist_var
            graphs_sampling_results[Trial_ID]['KS hop plots LCC Distr var'] = KS_hop_plots_LCC_dist_var
            pickle.dump(graphs_sampling_results, open(sampling_experiment_folder + 'sampling_results_v2', 'wb'))

    return graphs_sampling_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='My script')
    parser.add_argument('-root_folder', type=str, help='This is the .', )
    parser.add_argument('-dataset_folder', type=str, help='This is the number of dataset.')
    parser.add_argument('-graph_file_name', type=str, help='This is the number of dataset.')
    parser.add_argument('-sampling_setting_num', type=str, help='This is the sampling set')
    # Parse arguments
    args = parser.parse_args()
    root_folder = args.root_folder
    graph_file_name = args.graph_file_name
    sampling_setting_num = args.sampling_setting_num
    orig_graph_feature_folder = root_folder + '/features/' + graph_file_name + '/' 
    orig_graph_ID = graph_file_name[3:-7]
    orig_graph_folder = args.dataset_folder
    sample_graphs_folder = root_folder + '/samplings/setting_' + str(sampling_setting_num)  +  '/' + orig_graph_ID + '/sample_graphs/' 
    sampling_experiment_folder = root_folder + '/samplings/setting_' + str(sampling_setting_num) +  '/' + orig_graph_ID + '/'
    input_graph = load_graph(orig_graph_folder, graph_file_name)
    sample_graphs = load_sample_graphs(sample_graphs_folder)
    print('graph loaded')

    # Check whether the specified path exists or not
    isExist = os.path.exists(sampling_experiment_folder)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(sampling_experiment_folder)
        print("The new directory is created!")

    cc_interval = 0.001
    graph_sampling_results = evaluate_samplings(input_graph, sample_graphs)
    pickle.dump(graph_sampling_results, open(sampling_experiment_folder + 'sampling_results_v2', 'wb'))


