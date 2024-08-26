import snap
import pickle
import time
import argparse
import numpy as np
import os
from scipy.stats import entropy
from Graph_Processing_fast_v3 import graph_feature_extractor
from Calc_Distributions import calc_cdf, calc_normalized_cdf_continuous, calc_normalized_pdf_continuous

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
cc_interval = 0.001

#if not os.path.exists(feature_folder + graph_file_name + '_clustering_coeff_with_interval'): 
if True:
    graph = pickle.load(open(graph_folder + graph_file_name, 'rb'))

    gp = graph_feature_extractor()
    graph_snap = gp.convert_nx_to_snap_graph(graph)

    print('calculating clustering coeff')
    start_time = time.time()

    clust_coeff = graph_snap.GetNodeClustCfAll()
    end_time = time.time()
    clust_coeff_calc_time = end_time - start_time
    clust_coeff_list = [clust_coeff[i] for i in graph.nodes()]
    min_cc = min(clust_coeff_list)
    max_cc = max(clust_coeff_list)
    mean_cc = np.mean(clust_coeff_list)
    var_cc = np.var(clust_coeff_list)
    median_cc = np.median(clust_coeff_list)
    distr_cc = calc_normalized_cdf_continuous(clust_coeff_list, cc_interval = cc_interval)
    pdf_cc = calc_normalized_pdf_continuous(clust_coeff_list, cc_interval = cc_interval)
    gp.set_graph(graph)
    entropy_cc = entropy(pdf_cc)

    #clust_coeff_info = {'clust coeff': clust_coeff_list, 'calc time': clust_coeff_calc_time}
    clust_coeff_info = {'clust coeff distr': distr_cc, 'calc time': clust_coeff_calc_time, 'clust coeff min': min_cc, 'clust coeff max': max_cc, 'clust coeff mean': mean_cc, 'clust coeff var': var_cc, 'clust coeff median': median_cc, 'cc_interval': cc_interval, 'entropy_clust_coeff': entropy_cc}

    pickle.dump(clust_coeff_info, open(feature_folder + graph_file_name + '_clustering_coeff', 'wb'))
