import snap
import pickle
import time
import argparse
import json
import os
import numpy as np
import networkit as nk
from itertools import zip_longest
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
graph_nk = nk.nxadapter.nx2nk(graph)

# calc path features for all components
# component calc
start_t = time.time()
cc = nk.components.ConnectedComponents(graph_nk)
cc.run()
end_t = time.time()
connected_components_calc_time = end_t - start_t
cc_info = {'num cc': cc.numberOfComponents(), 'cc sizes': list(cc.getComponentSizes().values()) , 'cc calc time': connected_components_calc_time}

with open(feature_folder + graph_file_name + '_connected_comp.json', 'w') as fout:
    json.dump(cc_info, fout)

if not os.path.exists(feature_folder + graph_file_name + '_sp_lcc_stat_distr_approx_nk.json'):
  all_shortest_path_counts = []
  all_hop_plots = []
  max_comp_size = 0
  indx_largest_comp = -1
  idx = 0
  
  start_t = time.time()
  for comp in cc.getComponents():
    print(comp)
    comp_size = len(comp)
    #if comp_size == 1:
    #  idx += 1
    #  continue
    if comp_size > max_comp_size:
      max_comp_size = comp_size
      indx_largest_comp = idx
    comp_graph = nk.graphtools.subgraphFromNodes(graph_nk, comp)
    hp = nk.distance.HopPlotApproximation(comp_graph, k = 1024)
    hp.run()
    hop_plots = hp.getHopPlot()
    #print('hop plots comp ', hop_plots)
    node_pair_num = (comp_size * (comp_size - 1)) / 2 + comp_size
    hop_plots_list = [hop_plots[dist] * node_pair_num for dist in hop_plots.keys()]
    shortest_path_counts = [hop_plots_list[i+1] - hop_plots_list[i] for i in range(len(hop_plots_list) - 1)]
    shortest_path_counts.insert(0, hop_plots_list[0])
    #print('shortest path comp ', shortest_path_counts)
    all_shortest_path_counts.append(shortest_path_counts)
    all_hop_plots.append(hop_plots)
    idx += 1
  
  end_t = time.time()
  shortest_path_lengths_calc_time = start_t - end_t
  
  # LCC path features
  shortest_path_counts_LCC = all_shortest_path_counts[indx_largest_comp]
  len_shortest_path_list_lcc = len(shortest_path_counts_LCC)
  for i in range(len_shortest_path_list_lcc):
    if shortest_path_counts_LCC[-1] == 0:
      del shortest_path_counts_LCC[-1]
    else:
      break
  
  # calc statistics lcc paths
  sum_spl_counts_lcc = sum(shortest_path_counts_LCC)
  min_shortest_path_lcc = 0
  max_shortest_path_lcc = len(shortest_path_counts_LCC) - 1
  mean_shortest_path_lcc = sum([i * shortest_path_counts_LCC[i] for i in range(len(shortest_path_counts_LCC))])/ sum_spl_counts_lcc
  var_shortest_path_lcc = sum([shortest_path_counts_LCC[j] * (j - mean_shortest_path_lcc) ** 2  for j in range(len(shortest_path_counts_LCC))]) / sum_spl_counts_lcc
  # calc hop plot lcc
  hop_plots_lcc = all_hop_plots[indx_largest_comp]
  # dic -> list
  hop_plots_lcc_list = [hop_plots_lcc[dist] for dist in hop_plots_lcc.keys()]
  # remove last extra 1s
  len_hop_plots_lcc_list = len(hop_plots_lcc_list)
  for i in range(len_hop_plots_lcc_list):
    if hop_plots_lcc_list[-1] == 1:
      del hop_plots_lcc_list[-1]
    else:
      break
  hop_plots_lcc_list.append(1)
  
  # save lcc path features
  sp_lcc_info = {'shortest paths lengths min': min_shortest_path_lcc, 'shortest paths lengths max': max_shortest_path_lcc, 'shortest paths lengths mean': mean_shortest_path_lcc, 'shortest paths lengths var': var_shortest_path_lcc, 'shortest paths lengths distr': hop_plots_lcc_list}
  
  with open(feature_folder + graph_file_name + '_sp_lcc_stat_distr_approx_nk.json', 'w') as fout:
      json.dump(sp_lcc_info, fout)


# all components path features
if not os.path.exists(feature_folder + graph_file_name + '_sp_stat_distr_approx_nk.json'):
  aggr_all_shortest_path_counts = []
  for comp_shortest_paths in all_shortest_path_counts:
    aggr_all_shortest_path_counts = [x + y for x,y in zip_longest(aggr_all_shortest_path_counts, comp_shortest_paths, fillvalue=0)]
  
  #print('all shortest paths ', aggr_all_shortest_path_counts)
  aggr_all_shortest_path_counts
  len_shortest_path_list = len(aggr_all_shortest_path_counts)
  for i in range(len_shortest_path_list):
    if aggr_all_shortest_path_counts[-1] == 0:
      del aggr_all_shortest_path_counts[-1]
    else:
      break
  
  # calc shortest path statistics
  sum_spl_counts = sum(aggr_all_shortest_path_counts)
  min_shortest_path = 0
  max_shortest_path = len(aggr_all_shortest_path_counts) - 1
  mean_shortest_path = sum([i * aggr_all_shortest_path_counts[i] for i in range(len(aggr_all_shortest_path_counts))])/ sum_spl_counts
  var_shortest_path = sum([aggr_all_shortest_path_counts[j] * (j - mean_shortest_path) ** 2 for j in range(len(aggr_all_shortest_path_counts))]) / sum_spl_counts
  # calc hop plot aggregated
  hop_plots_aggr_list = list(np.cumsum(aggr_all_shortest_path_counts) / sum_spl_counts)
  
  # save aggr path features
  sp_info = {'shortest paths lengths min': min_shortest_path, 'shortest paths lengths max': max_shortest_path, 'shortest paths lengths mean': mean_shortest_path, 'shortest paths lengths var': var_shortest_path, 'shortest paths lengths distr': hop_plots_aggr_list, 'shortest_path_lengths_calc_time': shortest_path_lengths_calc_time}
  
  with open(feature_folder + graph_file_name + '_sp_stat_distr_approx_nk.json', 'w') as fout:
      json.dump(sp_info, fout)


