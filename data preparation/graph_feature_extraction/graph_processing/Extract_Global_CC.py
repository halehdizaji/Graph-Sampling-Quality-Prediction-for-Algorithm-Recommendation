import snap
import pickle
import time
import argparse
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

if not os.path.exists(feature_folder + graph_file_name + '_global_clustering_coeff'):
    graph = pickle.load(open(graph_folder + graph_file_name, 'rb'))

    gp = graph_feature_extractor()
    gp.set_graph(graph)
    print('calculating graph degrees')
    gp.degrees = gp.graph.degree()
    gp.degrees_list = [gp.degrees[i] for i in gp.graph.nodes()]

    print('calculating global clustering coefficient')
    start_time = time.time()
    gp.calc_global_clust_coeff()
    #self.calc_global_clust_coeff_cnx()
    end_time = time.time()
    gp.global_clust_coeff_calc_time = end_time - start_time

    gp.global_clust_coeff
    global_clust_coeff_info = {'global clust coeff': gp.global_clust_coeff, 'calc time': gp.global_clust_coeff_calc_time}

    pickle.dump(global_clust_coeff_info, open(feature_folder + graph_file_name + '_global_clustering_coeff', 'wb'))
