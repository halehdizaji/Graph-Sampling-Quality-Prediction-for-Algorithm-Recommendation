import snap
import pickle
import time
import argparse
import json
import numpy as np
from Graph_Processing_fast_v3 import graph_feature_extractor
from Calc_Distributions import calc_cdf

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
gp.set_graph(graph)
graph_snap = gp.convert_nx_to_snap_graph(graph)

print('calculating connected components')
start_time = time.time()
gp.calc_connected_components()
end_time = time.time()
gp.connected_components_calc_time = end_time - start_time
gp.number_connected_components = gp.num_connected_components()
gp.calc_connected_components_sizes()
#self.num_connected_components = cnx.number_connected_components(self.graph)
end_time = time.time()
print('running of non-threaded functions is ', end_time - start_time)

cc_info = {'num cc': gp.number_connected_components, 'cc sizes': gp.connected_components_sizes , 'cc calc time': gp.connected_components_calc_time, }

with open(feature_folder + graph_file_name + '_connected_comp.json', 'w') as fout:
    json.dump(cc_info, fout)

############################### calculating diameter of the largest component #####
#print([g for g in self.connected_components])

largest_cc = max(gp.connected_components_nodes, key=len)
largest_cc_graph = gp.graph.subgraph(largest_cc).copy()
'''
start_time = time.time()
gp.diameter_largest_cc = gp.diameter_component(largest_cc_graph)
end_time = time.time()
gp.diameter_calc_time = end_time - start_time
'''
####################################### Eccentricity ###########################
largest_cc_graph_snap = gp.convert_nx_to_snap_graph(largest_cc_graph)
'''
gp.eccentricity_centrality_LCC_list = []
print('calculating eccentricity centrality')
start_time = time.time()
for node in largest_cc_graph_snap.Nodes():
    gp.eccentricity_centrality_LCC_list.append(largest_cc_graph_snap.GetNodeEcc(node.GetId(), True))
end_time = time.time()
#print('eccenticities: ', self.eccentricity_centrality_LCC_list)
gp.eccentricity_centrality_LCC_calc_time = end_time - start_time

ecc_info = {'ecc list': gp.eccentricity_centrality_LCC_list, 'ecc calc time': gp.eccentricity_centrality_LCC_calc_time}

pickle.dump(ecc_info, open(feature_folder + graph_file_name + '_ecc_LCC', 'wb'))
'''
##########################################
print('calculating shortest paths for LCC')

gp.shortest_path_lengths = []
ecc_list = []
fc_list = []
lcc_size = len(largest_cc)
start_time = time.time()
for node in largest_cc_graph_snap.Nodes():
    _, NIdToDistH = largest_cc_graph_snap.GetShortPathAll(node.GetId())
    #print('number of node ', len(NIdToDistH.keys()))
    #print(NIdToDistH)
    sum_dist_node = 0
    max_dist_node = 0
    for item in NIdToDistH:
        if item >= node.GetId():
            dist = NIdToDistH[item]
            gp.shortest_path_lengths.append(dist)
            #nodes_pairs.append((node.GetId(), item))
            sum_dist_node += dist
            if dist > max_dist_node:
                max_dist_node = dist
    mean_dist_node = sum_dist_node/(lcc_size - 1)
    fc_list.append(mean_dist_node)
    ecc_list.append(max_dist_node)

print('calculating shortest paths for other components')

# calc spl for other components
gp.connected_components_nodes.remove(largest_cc)

for comp in gp.connected_components_nodes:
    cc_graph = gp.graph.subgraph(comp).copy()
    cc_graph_snap = gp.convert_nx_to_snap_graph(cc_graph)
    for node in cc_graph_snap.Nodes():
        _, NIdToDistH = cc_graph_snap.GetShortPathAll(node.GetId())
        for item in NIdToDistH:
            if item >= node.GetId():
                gp.shortest_path_lengths.append(NIdToDistH[item])
    
end_time = time.time()
gp.shortest_path_lengths_calc_time = end_time - start_time
min_sp = min(gp.shortest_path_lengths)
max_sp = max(gp.shortest_path_lengths)
mean_sp = np.mean(gp.shortest_path_lengths)
var_sp = np.var(gp.shortest_path_lengths)
median_sp = np.median(gp.shortest_path_lengths)
distr_sp = calc_cdf(gp.shortest_path_lengths)

sp_info = {'shortest paths lengths min': min_sp, 'shortest paths lengths max': max_sp, 'shortest paths lengths mean': mean_sp, 'shortest paths lengths median': median_sp, 'shortest paths lengths var': var_sp, 'shortest paths lengths distr': distr_sp, 'shortest_path_lengths_calc_time': gp.shortest_path_lengths_calc_time}

with open(feature_folder + graph_file_name + '_sp.json', 'w') as fout:
    json.dump(sp_info, fout)

