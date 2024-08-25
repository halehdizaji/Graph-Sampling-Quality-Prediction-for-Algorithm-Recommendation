################################# In this file graphs are saved with the name containing size number and an ID for each generated graph ###############################
############################## In this file also settings are read from the file.
import random
import networkx as nx
import pickle
import os
import configparser
import ast
import datetime
import sys
sys.path.append('../')
from Graph_Processing.Graph_Processing_fast_v3 import graph_feature_extractor
from Data_Preprocessing.Read_Graph_from_Text import read_graph_from_text, read_graph_from_text_no_self_loop, read_graph_from_text_pd_no_self_loop
from Text_Processing.Read_Config import read_config_file_synthetic_data

current_datetime = datetime.datetime.now()
current_datetime_str = str(current_datetime.year) + str(current_datetime.month) + str(current_datetime.day) + str(current_datetime.hour)
dataset_num = 2 
data_folder = '/home/itec/haleh/PycharmProjects/Graph_Sampling_Alg_Selection_Small_Scale/venv/data/graphs_data/real_graphs/dataset_' + str(dataset_num) + '/'
graphs_folder = data_folder + 'graphs/'
txt_files_floder = data_folder + 'txt_files/'

###################################################################################
real_train_txt_graphs_folder = txt_files_floder +'/train/'
real_train_graphs_folder = graphs_folder + 'train/'

real_test_txt_graphs_folder = txt_files_floder +'/test/'
real_test_graphs_folder =  graphs_folder + 'test/'


# Check whether the specified path exists or not
isExist = os.path.exists(real_train_graphs_folder)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(real_train_graphs_folder)
   print("The new directory is created!")

# Check whether the specified path exists or not
isExist = os.path.exists(real_test_graphs_folder)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(real_test_graphs_folder)
   print("The new directory is created!")

graphs_info = {}

def const_real_graphs(real_data_folder, data_type = 'train'):
    if data_type == 'train':
        real_graphs_folder = real_train_graphs_folder
    else:
        real_graphs_folder = real_test_graphs_folder

    for entry in os.listdir(real_data_folder):
        if os.path.isfile(os.path.join(real_data_folder, entry)):
            filepath = real_data_folder + entry
            #graph = read_graph_from_text(filepath)
            graph = read_graph_from_text_pd_no_self_loop(filepath)
            gp = graph_feature_extractor()
            gp.set_graph(graph)
            gp.calc_node_nums()
            graph_density = gp.graph_density()
            print('graph density ', graph_density)
            graph_ID = data_type + '_Real_' + entry
            pickle.dump(graph, open(
                real_graphs_folder + 'ID_' + str(graph_ID) + '.pickle', 'wb'))
            # print(g_ER.edges())
            print('one erdos renyi graph containing ' + str(gp.node_nums) + ' nodes generated.')
            graphs_info[graph_ID] = {'graph_type': 'real_' + entry, 'graph_node_nums': gp.node_nums,
                                     'graph_density': graph_density, 'graph_param': None}
    return

const_real_graphs(real_train_txt_graphs_folder, data_type='train')
const_real_graphs(real_test_txt_graphs_folder, data_type='test')

pickle.dump(graphs_info, open(data_folder + 'graphs_info.pkl', 'wb'))
