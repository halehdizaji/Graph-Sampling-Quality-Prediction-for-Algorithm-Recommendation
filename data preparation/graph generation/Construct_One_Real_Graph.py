################################# In this file graphs are saved with the name containing size number and an ID for each generated graph ###############################
############################## In this file also settings are read from the file.

import pickle
import os
import sys
sys.path.append('../')
from Graph_Processing.Graph_Processing import graph_feature_extractor
from Data_Preprocessing.Read_Graph_from_Text import read_graph_from_text, read_graph_from_csv_ids, read_graph_from_orig_csv_ids, read_graph_from_text_pd_no_self_loop
from Text_Processing.Read_Config import read_config_file_synthetic_data

dataset_num = 2
data_folder = '/home/haleh/PycharmProjects/Graph_Sampling_Alg_Selection/venv/data/graphs_data/real_graphs/dataset_' + str(dataset_num) + '/'
graphs_folder = data_folder + 'graphs/'
txt_files_floder = data_folder + 'txt_files/'
#graph_file_name = 'Cit-HepTh.csv'
#graph_file_name = 'Internet_Topology.csv'
graph_file_name = 'out.cit-HepPh'
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

def const_real_graph(real_data_folder, file_name, data_type = 'train'):
    if data_type == 'train':
        real_graphs_folder = real_train_graphs_folder
    else:
        real_graphs_folder = real_test_graphs_folder

    if os.path.isfile(os.path.join(real_data_folder, file_name)):
        filepath = real_data_folder + file_name
        print(filepath)
        ################### read from txt file ##################
        #graph = read_graph_from_text(filepath)
        ##################### read from csv file #################
        #graph = read_graph_from_text_pd_no_self_loop(filepath)
        ##################### read from csv file with ids #################
        graph = read_graph_from_orig_csv_ids(filepath, start_raw = 1)
        gp = graph_feature_extractor()
        gp.set_graph(graph)
        gp.calc_node_nums()
        graph_density = gp.graph_density()
        print('graph density ', graph_density)
        print(graph.edges())
        print('num edges: ', len(graph.edges()))
        graph_ID = data_type + '_Real_' + file_name
        pickle.dump(graph, open(
            real_graphs_folder + 'ID_' + str(graph_ID) + '.pickle', 'wb'))
        # print(g_ER.edges())
        print('one erdos renyi graph containing ' + str(gp.node_nums) + ' nodes generated.')
        graphs_info[graph_ID] = {'graph_type': 'real_' + file_name, 'graph_node_nums': gp.node_nums,
                                 'graph_density': graph_density, 'graph_param': None}
    return

#const_real_graph(real_train_txt_graphs_folder, data_type='train')
const_real_graph(real_test_txt_graphs_folder, graph_file_name, data_type='test')

pickle.dump(graphs_info, open(data_folder + graph_file_name + '_graph_info.pkl', 'wb'))
