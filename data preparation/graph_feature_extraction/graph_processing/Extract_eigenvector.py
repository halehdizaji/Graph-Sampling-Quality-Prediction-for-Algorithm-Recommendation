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
if not os.path.exists(feature_folder + graph_file_name + '_eigenvector'):
    graph = pickle.load(open(graph_folder + graph_file_name, 'rb'))

    gp = graph_feature_extractor()
    gp.set_graph(graph)

    print('calculating eigenvector centrality ')
    start_time = time.time()
    ######## nx
    gp.eigenvector_centrality = nx.eigenvector_centrality(gp.graph, max_iter=300, tol=1.0e-3)
    end_time = time.time()
    gp.eigenvector_centrality_calc_time = end_time - start_time
    gp.eigenvector_centrality_list = [gp.eigenvector_centrality[i] for i in gp.graph.nodes()]
    min_ev = min(gp.eigenvector_centrality_list)
    max_ev = max(gp.eigenvector_centrality_list)
    mean_ev = np.mean(gp.eigenvector_centrality_list)
    var_ev = np.var(gp.eigenvector_centrality_list)
    median_ev = np.median(gp.eigenvector_centrality_list)

    #ev_info = {'EV': gp.eigenvector_centrality_list, 'calc time': gp.eigenvector_centrality_calc_time}
    ev_info = {'min EV': min_ev, 'max EV': max_ev, 'mean EV': mean_ev, 'var EV': var_ev, 'median EV': median_ev, 'calc time': gp.eigenvector_centrality_calc_time}
    pickle.dump(ev_info, open(feature_folder + graph_file_name + '_eigenvector', 'wb'))
