import snap
import pickle
import time
import argparse
import networkx as nx
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

if not os.path.exists(feature_folder + graph_file_name + '_assortativity'):
    graph = pickle.load(open(graph_folder + graph_file_name, 'rb'))

    gp = graph_feature_extractor()
    gp.set_graph(graph)
    start_time = time.time()
    gp.degree_assortativity = nx.degree_assortativity_coefficient(gp.graph)
    end_time = time.time()
    gp.degree_assortativity_calc_time = end_time - start_time

    pr_info = {'degree_assortativity': gp.degree_assortativity, 'degree_assortativity_calc_time': gp.degree_assortativity_calc_time}
    pickle.dump(pr_info, open(feature_folder + graph_file_name + '_assortativity', 'wb'))
