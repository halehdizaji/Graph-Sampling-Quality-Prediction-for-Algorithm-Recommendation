import snap
import pickle
import time
import argparse
import json
import numpy as np
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

graph = pickle.load(open(graph_folder + graph_file_name, 'rb'))

gp = graph_feature_extractor()
graph_snap = gp.convert_nx_to_snap_graph(graph)

start_t = time.time()
print('started betweenness calc ')
bc_accuracy = 0.8
Nodes, Edges = graph_snap.GetBetweennessCentr(bc_accuracy)
end_t = time.time()

calc_time =  end_t - start_t
node_betweenness_centrality_list = [Nodes[i] for i in Nodes]
edge_betweenness_centrality_list = [Edges[i] for i in Edges]
min_edge_betweenness = min(edge_betweenness_centrality_list)
max_edge_betweenness = max(edge_betweenness_centrality_list)
mean_edge_betweenness = np.mean(edge_betweenness_centrality_list)
var_edge_betweenness = np.var(edge_betweenness_centrality_list)
median_edge_betweenness = np.median(edge_betweenness_centrality_list)

min_node_betweenness = min(node_betweenness_centrality_list)
max_node_betweenness = max(node_betweenness_centrality_list)
mean_node_betweenness = np.mean(node_betweenness_centrality_list)
var_node_betweenness = np.var(node_betweenness_centrality_list)
median_node_betweenness = np.median(node_betweenness_centrality_list)

#betweenness_info = {'edge_betweenness_centrality_list': edge_betweenness_centrality_list, 'node_betweenness_centrality_list': node_betweenness_centrality_list, 'calc time': calc_time}
betweenness_info = {'min_edge_betweenness': min_edge_betweenness, 'max_edge_betweenness': max_edge_betweenness, 'mean_edge_betweenness': mean_edge_betweenness, 'var_edge_betweenness': var_edge_betweenness, 'median_edge_betweenness': median_edge_betweenness, 'min_node_betweenness':min_node_betweenness, 'max_node_betweenness': max_node_betweenness, 'mean_node_betweenness':mean_node_betweenness, 'var_node_betweenness': var_node_betweenness, 'median_node_betweenness':median_node_betweenness, 'calc time': calc_time}

with open(feature_folder + graph_file_name + '_betweenness_' + str(bc_accuracy) + '.json', 'w') as fout:
    json.dump(betweenness_info, fout)

