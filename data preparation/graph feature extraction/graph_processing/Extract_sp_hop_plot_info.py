import snap
import pickle
import time
import argparse
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
gp.set_graph(graph)
graph_snap = gp.convert_nx_to_snap_graph(graph)

############################# load shortest paths LCC info ##################
sp_LCC_info = pickle.load(open(feature_folder + graph_file_name + '_sp_LCC', 'rb'))
sp_LCC_list = sp_LCC_info['shortest paths lengths']
sp_LCC_calc_time = sp_LCC_info['shortest_path_lengths_LCC_calc_time']

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


############################### calculating diameter of the largest component #####
#print([g for g in self.connected_components])

largest_cc = max(gp.connected_components_nodes, key=len)
gp.connected_components_nodes.remove(largest_cc)

gp.shortest_path_lengths = []
for component in gp.connected_components_nodes:
    comp_graph = graph.subgraph(component)
    cc_graph_snap = gp.convert_nx_to_snap_graph(comp_graph)

    for node in cc_graph_snap.Nodes():
        _, NIdToDistH = cc_graph_snap.GetShortPathAll(node.GetId())
        #print('number of node ', len(NIdToDistH.keys()))
        #print(NIdToDistH)
        for item in NIdToDistH:
            if item >= node.GetId():
                gp.shortest_path_lengths.append(NIdToDistH[item])
                #nodes_pairs.append((node.GetId(), item))
                #print(item, NIdToDistH[item])

sp_info = {'shortest paths lengths': gp.shortest_path_lengths, 'shortest_path_lengths_LCC_calc_time': gp.shortest_path_lengths_LCC_calc_time}

pickle.dump(sp_info, open(feature_folder + graph_file_name + '_sp_LCC', 'wb'))

##################################

gp.farness_centrality_list = []
print('calculating farness centrality')
start_time = time.time()
for node in largest_cc_graph_snap.Nodes():
    gp.farness_centrality_list.append(largest_cc_graph_snap.GetFarnessCentr(node.GetId(), True))
end_time = time.time()
gp.farness_centrality_calc_time = end_time - start_time

farness_info = { 'farness': gp.farness_centrality_list, 'farness calc time': gp.farness_centrality_calc_time}

pickle.dump(farness_info, open(feature_folder + graph_file_name + '_farness_LCC', 'wb'))

