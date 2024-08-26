################################# In this file graphs are saved with the name containing size number and an ID for each generated graph ###############################
############################## In this file also settings are read from the file.
import random
import networkx as nx
import pickle
import os
import argparse
import ast
import datetime
import numpy as np
#Simport snap
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from Graph_Processing.Graph_Processing_fast_v3 import graph_feature_extractor
from Graph_Processing.Graph_Functions import approximate_albert_barabasi_param
from Text_Preprocessing.Read_Graph_from_Text import read_graph_from_text
from Text_Processing.Read_Config import read_config_file_synthetic_data_v4

current_datetime = datetime.datetime.now()
current_datetime_str = str(current_datetime.year) + str(current_datetime.month) + str(current_datetime.day) + str(current_datetime.hour)
# Create argument parser
parser = argparse.ArgumentParser(description='My script')

# Add arguments
parser.add_argument('-dataset_num', type=str, help='This is the number of dataset.', )

# Parse arguments
args = parser.parse_args()

# Parse arguments
dataset_num = args.dataset_num

data_folder = './Graph_Sampling_Alg_Selection_Small_Scale/venv/data/graphs_data/generated_graphs/dataset_' + str(dataset_num) + '/'
config_file_path = data_folder + 'dataset_info'
synthetic_train_graphs, train_generated_graph_types, train_input_num_per_type_size, \
    train_graphs_sizes, train_graphs_sizes_ranges, train_generated_graphs_densities, train_stochastic_block_model_cluster_nums, train_stochastic_block_model_probs_ratios, train_powerlaw_cluster_params, train_forest_fire_probs = read_config_file_synthetic_data_v4(config_file_path, 'train')
synthetic_test_graphs, test_generated_graph_types, test_input_num_per_type_size, \
    test_graphs_sizes, test_graphs_sizes_ranges, test_generated_graphs_densities, test_stochastic_block_model_cluster_nums, test_stochastic_block_model_probs_ratios, test_powerlaw_cluster_params, test_forest_fire_probs = read_config_file_synthetic_data_v4(config_file_path, 'test')

###################################################################################
generated_train_graphs_folder = data_folder + 'train/'
generated_test_graphs_folder = data_folder + 'test/'

# Check whether the specified path exists or not
isExist = os.path.exists(generated_train_graphs_folder)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(generated_train_graphs_folder)
   print("The new directory is created!")

# Check whether the specified path exists or not
isExist = os.path.exists(generated_test_graphs_folder)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(generated_test_graphs_folder)
   print("The new directory is created!")

graphs_info = {}

def generate_graphs(data_type= 'train'):
    if data_type == 'train':
        print('data type ', data_type)
        generated_graph_types = train_generated_graph_types
        graphs_sizes = train_graphs_sizes
        graphs_sizes_ranges = train_graphs_sizes_ranges
        input_num_per_type_size = train_input_num_per_type_size
        generated_graphs_densities = train_generated_graphs_densities
        generated_graphs_folder = generated_train_graphs_folder
        stochastic_block_model_cluster_nums = train_stochastic_block_model_cluster_nums
        stochastic_block_model_probs_ratios = train_stochastic_block_model_probs_ratios
        powerlaw_cluster_params = train_powerlaw_cluster_params
        forest_fire_probs = train_forest_fire_probs

    else:
        generated_graph_types = test_generated_graph_types
        graphs_sizes = test_graphs_sizes
        graphs_sizes_ranges = test_graphs_sizes_ranges
        input_num_per_type_size = test_input_num_per_type_size
        generated_graphs_densities = test_generated_graphs_densities
        generated_graphs_folder = generated_test_graphs_folder
        stochastic_block_model_cluster_nums = test_stochastic_block_model_cluster_nums
        stochastic_block_model_probs_ratios = test_stochastic_block_model_probs_ratios
        powerlaw_cluster_params = test_powerlaw_cluster_params
        forest_fire_probs = test_forest_fire_probs

    if generated_graph_types['albert_barabasi']:
        size_indx = 1
        for key in graphs_sizes.keys():
            if graphs_sizes[key]:
                start_indx = graphs_sizes_ranges[key][0]
                end_indx = graphs_sizes_ranges[key][1]
                for j in range(input_num_per_type_size):
                    node_nums = random.randint(start_indx, end_indx)
                    for graph_density in generated_graphs_densities:
                        # param_ba = random.randint(1, 2)
                        # param_ba = 1
                        graph_density = float(graph_density)
                        print('density ', graph_density)
                        param_ba = approximate_albert_barabasi_param(node_nums, graph_density)
                        if param_ba == 0:
                            continue
                        g_ba = nx.barabasi_albert_graph(node_nums, param_ba)
                        # nx.draw_kamada_kawai(g_ba)
                        # nx.draw_networkx(g_ba)
                        gp = graph_feature_extractor()
                        gp.set_graph(g_ba)
                        gp.calc_node_nums()
                        print('node nums: ', gp.node_nums)
                        graph_density = gp.graph_density()
                        print('graph density ', graph_density)
                        graph_ID = data_type + '_Syn_albert_barabasi_' + key + '_param:'+  str(param_ba) + '_' + str(j) + '_' + current_datetime_str
                        pickle.dump(g_ba, open(generated_graphs_folder + 'ID_' + str(
                            graph_ID) + '.pickle', 'wb'))
                        # print(g_ba.edges())
                        print('one albert barabasi graph containing ' + str(node_nums) + ' nodes generated.')
                        graphs_info[graph_ID] = {'graph_type': 'albert_barabasi', 'graph_node_nums': node_nums,
                                                 'graph_density': graph_density, 'graph_param': param_ba}
            size_indx += 1

    ###### The other models need correction.
    if generated_graph_types['watts_strogatz']:
        size_indx = 1
        for key in graphs_sizes.keys():
            if graphs_sizes[key]:
                start_indx = graphs_sizes_ranges[key][0]
                end_indx = graphs_sizes_ranges[key][1]
                for j in range(input_num_per_type_size):
                    node_nums = random.randint(start_indx, end_indx)
                    for graph_density in generated_graphs_densities:
                        # param_watts = 0.01 * random.randint(1, 3)
                        # param_watts = 0.02
                        graph_density = float(graph_density)
                        print('density ', graph_density)
                        watts_neighbour_num = round(graph_density * (node_nums - 1))
                        print('watts strogatz neighbor number is: ', watts_neighbour_num)
                        if watts_neighbour_num < 2:
                            continue
                        g_ws = nx.watts_strogatz_graph(node_nums, watts_neighbour_num, 0.4)
                        # nx.draw(g_wsg)
                        gp = graph_feature_extractor()
                        gp.set_graph(g_ws)
                        gp.calc_node_nums()
                        print('node nums: ', gp.node_nums)
                        graph_density = gp.graph_density()
                        print('graph density ', graph_density)
                        graph_ID = data_type + '_Syn_watts_strogatz_' + key + '_param:' + str(watts_neighbour_num) + '_' + str(j) + current_datetime_str
                        # pickle.dump(g_ws, open('./data/watts_strogatz_size'+ str(size_indx) + '_g' + str(i + start_indx_input_num) + '.pickle', 'wb'))
                        pickle.dump(g_ws, open(generated_graphs_folder + 'ID_' + str(
                            graph_ID) + '.pickle', 'wb'))
                        # print(g_ws.edges())
                        print('one watts strogatz graph containing ' + str(node_nums) + ' nodes generated.')
                        graphs_info[graph_ID] = {'graph_type': 'watts_strogatz', 'graph_node_nums': node_nums,
                                                 'graph_density': graph_density, 'graph_param': watts_neighbour_num}
            size_indx += 1

    if generated_graph_types['erdos_renyi']:
        size_indx = 1
        for key in graphs_sizes.keys():
            if graphs_sizes[key]:
                start_indx = graphs_sizes_ranges[key][0]
                end_indx = graphs_sizes_ranges[key][1]
                for j in range(input_num_per_type_size):
                    node_nums = random.randint(start_indx, end_indx)
                    for graph_density in generated_graphs_densities:
                        # param_ER_graphs = 0.0001 * random.randint(5, 10)
                        # ER_param = 0.05
                        graph_density = float(graph_density)
                        ER_param = graph_density
                        g_ER = nx.erdos_renyi_graph(node_nums, ER_param)
                        gp = graph_feature_extractor()
                        gp.set_graph(g_ER)
                        gp.calc_node_nums()
                        print('node nums: ', gp.node_nums)
                        graph_density = gp.graph_density()
                        print('graph density ', graph_density)
                        graph_ID = data_type + '_Syn_erdos_renyi_' + key + '_param:' + str(ER_param) + '_' + str(j) + current_datetime_str
                        # pickle.dump(g_ER, open('./data/erdos_renyi_size'+ str(size_indx) + '_g' + str(i + start_indx_input_num) + '.pickle', 'wb'))
                        pickle.dump(g_ER, open(generated_graphs_folder + 'ID_' + str(
                            graph_ID) + '.pickle', 'wb'))
                        # print(g_ER.edges())
                        print('one erdos renyi graph containing ' + str(node_nums) + ' nodes generated.')
                        graphs_info[graph_ID] = {'graph_type': 'erdos_renyi', 'graph_node_nums': node_nums,
                                                 'graph_density': graph_density, 'graph_param': ER_param}
            size_indx += 1

    if generated_graph_types['newman_watts_strogatz']:
        # needs to be edited
        size_indx = 1
        for key in graphs_sizes.keys():
            if graphs_sizes[key]:
                start_indx = graphs_sizes_ranges[key][0]
                end_indx = graphs_sizes_ranges[key][1]
                for j in range(input_num_per_type_size):
                    node_nums = random.randint(start_indx, end_indx)
                    for NWS_param in graphs_params['newman_watts_strogatz']:
                        # param_ER_graphs = 0.0001 * random.randint(5, 10)
                        # ER_param = 0.05
                        g_NWS = nx.newman_watts_strogatz_graph(node_nums, 2, NWS_param)
                        gp = graph_feature_extractor()
                        gp.set_graph(g_NWS)
                        gp.calc_node_nums()
                        graph_density = gp.graph_density()
                        print('graph density ', graph_density)
                        graph_ID = data_type + '_Syn_newman_ws_' + key + '_param:' + str(NWS_param) + '_' + str(j) + current_datetime_str
                        pickle.dump(g_NWS, open(generated_graphs_folder + 'ID_' + str(
                            graph_ID) + '.pickle', 'wb'))
                        # pickle.dump(g_ER, open('./data/erdos_renyi_size'+ str(size_indx) + '_g' + str(i + start_indx_input_num) + '.pickle', 'wb'))
                        # print(g_ER.edges())
                        print('one newman_watts_strogatz graph containing ' + str(node_nums) + ' nodes generated.')
                        graphs_info[graph_ID] = {'graph_type': 'newman_watts_strogatz', 'graph_node_nums': node_nums,
                                                 'graph_density': graph_density, 'graph_param': NWS_param}
            size_indx += 1

    if generated_graph_types['powerlaw_cluster']:
        # needs to be edited
        size_indx = 1
        for key in graphs_sizes.keys():
            if graphs_sizes[key]:
                start_indx = graphs_sizes_ranges[key][0]
                end_indx = graphs_sizes_ranges[key][1]
                for j in range(input_num_per_type_size):
                    node_nums = random.randint(start_indx, end_indx)
                    ms_tri_probs =powerlaw_cluster_params[key]
                    for m_tri_probs in ms_tri_probs:
                    	param_m = m_tri_probs[0]
                    	for prob in m_tri_probs[1]:
                            g_PLC = nx.powerlaw_cluster_graph(node_nums, param_m, prob)
                            gp = graph_feature_extractor()
                            gp.set_graph(g_PLC)
                            gp.calc_node_nums()
                            graph_density = gp.graph_density()
                            print('graph density ', graph_density)
                            # pickle.dump(g_ER, open('./data/erdos_renyi_size'+ str(size_indx) + '_g' + str(i + start_indx_input_num) + '.pickle', 'wb'))
                            graph_ID = data_type + '_Syn_powerlow_clust_' + key + '_params:m_' + str(param_m) + '_prob_' + str(prob) + '_' + str(j) + current_datetime_str
                            pickle.dump(g_PLC, open(generated_graphs_folder + 'ID_' + str(
                            graph_ID) + '.pickle', 'wb'))
                            # print(g_ER.edges())
                            print('one epowerlaw_cluster graph containing ' + str(node_nums) + ' nodes generated.')
                            graphs_info[graph_ID] = {'graph_type': 'powerlaw_cluster', 'graph_node_nums': node_nums,
                                                 'graph_density': graph_density, 'param_m': param_m, 'param_triad_prob': prob}
            size_indx += 1

    if generated_graph_types['stochastic_block_model']:
        # needs to be edited
        size_indx = 1
        for key in graphs_sizes.keys():
            if graphs_sizes[key]:
                start_indx = graphs_sizes_ranges[key][0]
                end_indx = graphs_sizes_ranges[key][1]
                for j in range(input_num_per_type_size):
                    node_nums = random.randint(start_indx, end_indx)
                    for density in generated_graphs_densities:
                        for cluster_num in stochastic_block_model_cluster_nums:
                            for stochastic_block_model_probs_ratio in stochastic_block_model_probs_ratios:
                                intra_cluster_density = (density * (node_nums-1) * cluster_num) / (node_nums - cluster_num + stochastic_block_model_probs_ratio * node_nums / (cluster_num - 2))
                                inter_cluster_density = stochastic_block_model_probs_ratio * intra_cluster_density
                                cluster_size = int(node_nums/cluster_num)
                                clusters_sizes = [cluster_size] * cluster_num
                                probs = inter_cluster_density * np.ones((cluster_num,cluster_num))
                                for i in range(cluster_num):
                                    probs[i,i] = intra_cluster_density
                                probs = probs.tolist()
                                g_sbm = nx.stochastic_block_model(clusters_sizes, probs)
                                gp = graph_feature_extractor()
                                gp.set_graph(g_sbm)
                                gp.calc_node_nums()
                                graph_density = gp.graph_density()
                                print('graph density ', graph_density)
                                # pickle.dump(g_ER, open('./data/erdos_renyi_size'+ str(size_indx) + '_g' + str(i + start_indx_input_num) + '.pickle', 'wb'))
                                graph_ID = data_type + '_Syn_SBM_' + key + '_params:cluster_nums_' + str(cluster_num) + '_density_' + str(density) + '_prob_ratio_' + str(stochastic_block_model_probs_ratio) + '_' + str(j) + current_datetime_str
                                pickle.dump(g_sbm, open(generated_graphs_folder + 'ID_' + str(
                                    graph_ID) + '.pickle', 'wb'))
                                print('one SBM graph containing ' + str(
                                    node_nums) + ' nodes generated.')
                                graphs_info[graph_ID] = {'graph_type': 'powerlaw_cluster',
                                                 		'graph_node_nums': node_nums,
                                                 		'graph_density': graph_density, 'param_clusters_sizes': clusters_sizes,
                                                 		'param_densities': probs}
		

            size_indx += 1

    if generated_graph_types['forest_fire']:
        # needs to be edited
        size_indx = 1
        for key in graphs_sizes.keys():
            if graphs_sizes[key]:
                start_indx = graphs_sizes_ranges[key][0]
                end_indx = graphs_sizes_ranges[key][1]
                for j in range(input_num_per_type_size):
                    node_nums = random.randint(start_indx, end_indx)

                    for prob in forest_fire_probs[key]:
                        g_ff = snap.GenForestFire(node_nums, prob, prob)
                        gp = graph_feature_extractor()
                        g_ff = gp.convert_snap_graph_to_nx(g_ff)
                        gp.set_graph(g_ff)
                        gp.calc_node_nums()
                        graph_density = gp.graph_density()
                        print('graph density ', graph_density)
                        # pickle.dump(g_ER, open('./data/erdos_renyi_size'+ str(size_indx) + '_g' + str(i + start_indx_input_num) + '.pickle', 'wb'))
                        graph_ID = data_type + '_Syn_FF_' + key + '_params_prob_' + str(prob) + '_' + str(j) + current_datetime_str
                        print(generated_graphs_folder + 'ID_' + str(
                            graph_ID) + '.pickle')
                        pickle.dump(g_ff, open(generated_graphs_folder + 'ID_' + str(
                            graph_ID) + '.pickle', 'wb'))
                        print('one FF graph containing ' + str(
                            node_nums) + ' nodes generated.')
                        graphs_info[graph_ID] = {'graph_type': 'forest_fire',
                                                 'graph_node_nums': node_nums,
                                                 'graph_density': graph_density, 'param_probs': prob}
                        #nx.draw(g_ff)


            size_indx += 1

    return

if synthetic_train_graphs:
    generate_graphs(data_type='train')
if synthetic_test_graphs:
    generate_graphs(data_type='test')

pickle.dump(graphs_info, open(data_folder + 'graphs_info.pkl', 'wb'))
