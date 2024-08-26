import snap
import pickle
import time
import argparse
import networkx as nx
import numpy as np
import os
from Graph_Processing_fast_v3 import graph_feature_extractor

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

# check for availability of feature and extract feature if doesn't exist
if not os.path.exists(feature_folder + graph_file_name + '_pagerank'):
    graph = pickle.load(open(graph_folder + graph_file_name, 'rb'))

    gp = graph_feature_extractor()
    gp.set_graph(graph)
    gp.snap_graph = gp.convert_nx_to_snap_graph(graph)

    start_time = time.time()
    gp.pagerank_centrality = gp.snap_graph.GetPageRank()
    end_time = time.time()
    gp.pagerank_centrality_calc_time = end_time - start_time

    ######
    gp.pagerank_centrality_list = [gp.pagerank_centrality[i] for i in gp.graph.nodes()]
    min_pr = min(gp.pagerank_centrality_list)
    max_pr = max(gp.pagerank_centrality_list)
    mean_pr = np.mean(gp.pagerank_centrality_list)
    var_pr = np.var(gp.pagerank_centrality_list)
    median_pr = np.median(gp.pagerank_centrality_list)

    pr_info = {'min PR': min_pr, 'max PR': max_pr, 'mean PR': mean_pr, 'median PR': median_pr, 'var_pr':var_pr, 'calc time': gp.pagerank_centrality_calc_time}
    pickle.dump(pr_info, open(feature_folder + graph_file_name + '_pagerank', 'wb'))
