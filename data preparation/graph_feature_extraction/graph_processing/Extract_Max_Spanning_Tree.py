import snap
import pickle
import time
import argparse
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

if not os.path.exists(feature_folder + graph_file_name + '_mst_deg'):
    graph = pickle.load(open(graph_folder + graph_file_name, 'rb'))

    gp = graph_feature_extractor()
    gp.set_graph(graph)
    print('calculating max spanning tree (or min spanning tree)')
    start_time = time.time()
    gp.calc_max_spanning_tree()
    end_time = time.time()
    gp.max_spanning_tree_calc_time = end_time - start_time
    gp.calc_degrees_max_spanning_tree()
    min_deg_mst = min(gp.max_spanning_tree_degrees)
    max_deg_mst = max(gp.max_spanning_tree_degrees)
    mean_deg_mst = np.mean(gp.max_spanning_tree_degrees)
    var_deg_mst = np.var(gp.max_spanning_tree_degrees)
    median_deg_mst = np.median(gp.max_spanning_tree_degrees)

    #mst_info = {'MST degrees': gp.max_spanning_tree_degrees, 'calc time': gp.max_spanning_tree_calc_time}
    mst_info = {'min_deg_mst': min_deg_mst, 'max_deg_mst': max_deg_mst, 'mean_deg_mst': mean_deg_mst, 'var_deg_mst': var_deg_mst, 'median_deg_mst': median_deg_mst, 'calc time': gp.max_spanning_tree_calc_time}
    pickle.dump(mst_info, open(feature_folder + graph_file_name + '_mst_deg', 'wb'))
