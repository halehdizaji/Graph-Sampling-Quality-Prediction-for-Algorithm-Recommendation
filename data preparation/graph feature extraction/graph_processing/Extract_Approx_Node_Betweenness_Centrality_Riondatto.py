### This script calculates approximate node betweenness centrality (BC) of a graph using method of Riondato as Networkit module.
# It returns statistics and distribution of normalized node BC values.

import os
import snap
import pickle
import time
import argparse
import json
import numpy as np
import networkit as nk
from Calc_Distributions import calc_normalized_cdf_continuous

parser = argparse.ArgumentParser(description='My script')

# Add arguments
parser.add_argument('-graph_folder', type=str, help='This is the number of feature set.', )
parser.add_argument('-graph_file_name', type=str, help='This is the number of dataset.')
parser.add_argument('-feature_folder', type=str, help='This is the folder of dataset.', )

# Parse arguments
args = parser.parse_args()
graph_folder = args.graph_folder
graph_file_name = args.graph_file_name
feature_folder = args.feature_folder

if not os.path.exists(feature_folder + graph_file_name + '_node_BC_approximate_riondato_stat_distr.json'):
    graph = pickle.load(open(graph_folder + graph_file_name, 'rb'))
    
    # convert nx graph to nk graph
    nk_graph = nk.nxadapter.nx2nk(graph)
    
    start_t = time.time()
    print('started betweenness calc ')
    bc_approx = nk.centrality.ApproxBetweenness(nk_graph)
    bc_approx.run()
    end_t = time.time()
    calc_time =  end_t - start_t
    
    node_betweenness_centrality_list = bc_approx.scores()
    min_node_betweenness = min(node_betweenness_centrality_list)
    max_node_betweenness = max(node_betweenness_centrality_list)
    mean_node_betweenness = np.mean(node_betweenness_centrality_list)
    var_node_betweenness = np.var(node_betweenness_centrality_list)
    median_node_betweenness = np.median(node_betweenness_centrality_list)
    
    # calc node BC distribution and save
    cdf_interval = 0.0001
    distr_nbc = calc_normalized_cdf_continuous(node_betweenness_centrality_list, cc_interval=cdf_interval)
    
    betweenness_info = {'BC cdf interval': cdf_interval, 'node BC distr': distr_nbc, 'min_node_betweenness':min_node_betweenness, 'max_node_betweenness': max_node_betweenness, 'mean_node_betweenness':mean_node_betweenness, 'var_node_betweenness': var_node_betweenness, 'median_node_betweenness':median_node_betweenness, 'calc time': calc_time}
    
    with open(feature_folder + graph_file_name + '_node_BC_approximate_riondato_stat_distr.json', 'w') as fout:
        json.dump(betweenness_info, fout)

