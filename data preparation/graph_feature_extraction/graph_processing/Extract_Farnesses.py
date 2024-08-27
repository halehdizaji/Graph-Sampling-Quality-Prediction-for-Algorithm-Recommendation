import snap
import pickle
import time
import sys
sys.path.append('../')
from Graph_Processing_fast_v3 import graph_feature_extractor

graph_folder = './Graph_Sampling_Alg_Selection/venv/data/graphs_data/real_graphs/dataset_1/graphs/test/'
graph_file_name = 'ID_test_Real_soc-wiki-Vote.mtx.pickle'
graph = pickle.load(open(graph_folder + graph_file_name, 'rb'))

gp = graph_feature_extractor()
graph_snap = gp.convert_nx_to_snap_graph(graph)
#UGraph = pickle.load(open(graph_folder + graph_file_name, 'rb'))
farness_file = graph_folder + 'temp/' + graph_file_name + '_farness'
#node_nums = 10000
#density = 0.2
#edge_num = int(density * (node_nums * (node_nums - 1))/2)
#UGraph = snap.GenRndGnm(snap.TUNGraph, node_nums, edge_num)
farness_centrality_list = []
print('calculating farness centrality')
start_time = time.time()
for node in graph_snap.Nodes():
    farness_centrality_list.append(graph_snap.GetFarnessCentr(node.GetId(), True))
end_time = time.time()
farness_centrality_calc_time = end_time - start_time
farness_info = {'farness list': farness_centrality_list, 'calc time': farness_centrality_calc_time}
pickle.dump(farness_info, open(farness_file, 'wb'))
