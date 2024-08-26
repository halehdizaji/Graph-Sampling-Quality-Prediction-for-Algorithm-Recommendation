import collections
import networkx as nx
import networkit as nk
import numpy as np
import torch
from itertools import zip_longest
import sys
sys.path.append('../')
#from Sampling_algorithms import node_sampling_random, node_sampling_degree
import time
import matplotlib.pyplot as plt
import pickle
import json
from scipy.stats import wasserstein_distance

def measure_avg_processing_time(graph, node_nums, time_constraint, sampling_algorithm, sampling_percent, iter= 1):
    avg_processing_time = 0
    if sampling_algorithm == "random_node":
        for i in range(iter):
            t_start = time.time()
            _ = node_sampling_random(graph, sampling_percent, node_nums)
            t_end = time.time()
            avg_processing_time += t_end - t_start
        avg_processing_time /= iter

    if sampling_algorithm == "random_node_degree":
        for i in range(iter):
            t_start = time.time()
            _ = node_sampling_degree(graph, sampling_percent, node_nums)
            t_end = time.time()
            avg_processing_time += t_end - t_start
        avg_processing_time /= iter

    return avg_processing_time


def measure_sampling_quality(graph, sampling_algorithm, iter= 1):

    ...


def calc_cdf(input_list):
    '''
        input:
            input_list: a list of values
        return:
            distribution of values in the input_list
        This function assumes 0 as the min value in the input_list and the max value in the list as the maximum value in the distribution.
    '''
    sorted_list = sorted(input_list)
    len_list = len(input_list)
    valueCount = collections.Counter(sorted_list)
    val, cnt = zip(*valueCount.items())
    val_list = list(val)
    #print('val list ', val_list)
    cnt_list = list(cnt)
    #print('count list ', cnt_list)
    cnt_all = [0] * (max(val_list) + 1)
    for i in range(len(val_list)):
        #print(' i is ', i)
        cnt_all[val_list[i]] = cnt_list[i]
    cs = np.cumsum(cnt_all)
    input_distr = cs / len_list
    return list(input_distr)


def measure_KS_D_statistics_graphs(graph, sampled_graph, desired_distribution):

    orig_node_nums = len(graph.nodes)
    sample_node_nums = len(sampled_graph.nodes)

    if desired_distribution == 'Degree':
        original_deg_dist = calc_normalized_degree_dist(graph, orig_node_nums)
        sample_deg_dist = calc_normalized_degree_dist(sampled_graph, sample_node_nums)
        KS_D_statistic_Degree = calc_KS_D_statistic(original_deg_dist, sample_deg_dist)
        return KS_D_statistic_Degree

    elif desired_distribution == "Clustering_Coeficient":
        original_CC_dist = calc_normalized_CC_dist(graph, orig_node_nums)
        sample_CC_dist = calc_normalized_CC_dist(sampled_graph, sample_node_nums)

    ...


def calc_normalized_degree_dist(graph, node_nums):
    sorted_degrees = sorted([d for n, d in graph.degree()])
    #print('sorted degrees ', sorted_degrees)
    degreeCount = collections.Counter(sorted_degrees)
    deg, cnt = zip(*degreeCount.items())
    deg_list = list(deg)
    cnt_list = list(cnt)
    #print('deg_list ', deg_list)
    #print('cnt_list ', cnt_list)
    #print('deg_list len ', len(deg_list))
    #print('cnt_list len ', len(cnt_list))
    #print('node nums ', node_nums)
    cnt_all = [0] * (node_nums + 2)
    for i in range(len(deg_list)):
        #print(' i is ', i)
        cnt_all[deg_list[i]] = cnt_list[i]
    cs = np.cumsum(cnt_all)
    degree_dist = cs / node_nums
    return list(degree_dist)


def calc_normalized_degree_dist_pdf(graph, node_nums):
    '''
    :param graph:
    :param node_nums:
    :return: returns normalized degree distribution of the graph
    '''
    sorted_degrees = sorted([d for n, d in graph.degree()])
    #print('sorted degrees ', sorted_degrees)
    degreeCount = collections.Counter(sorted_degrees)
    deg, cnt = zip(*degreeCount.items())
    deg_list = list(deg)
    cnt_list = list(cnt)
    #print('deg_list ', deg_list)
    #print('cnt_list ', cnt_list)
    #print('deg_list len ', len(deg_list))
    #print('cnt_list len ', len(cnt_list))
    #print('node nums ', node_nums)
    cnt_all = [0] * (node_nums + 2)
    for i in range(len(deg_list)):
        #print(' i is ', i)
        cnt_all[deg_list[i]] = cnt_list[i]
    normal_degree_dist_pdf = cnt_all / node_nums
    return normal_degree_dist_pdf


def calc_normalized_Clustering_Coeff_dist(graph, node_nums, cc_interval = 0.02):
    clust_coeff = nx.clustering(graph)
    print('finished calculating clustering coefficients')
    clust_coeff_sorted = sorted(clust_coeff.values())
    clust_coeff_dist = []
    #cc_list_len = int(1/cc_interval)

    start_interval = 0
    end_interval = start_interval + cc_interval
    count_interval = 0

    for cc in clust_coeff_sorted:
        if cc <= end_interval:
            count_interval += 1
        else:
            clust_coeff_dist.append(count_interval)
            count_interval = 1
            end_interval += cc_interval

    clust_coeff_dist.append(count_interval)
    #obtained_cc_list_len = len(clust_coeff_dist)
    #cc_list_diff = cc_list_len - obtained_cc_list_len

    #if obtained_cc_list_len < cc_list_len:
    #    clust_coeff_dist.extend([0] * cc_list_diff)

    cs = np.cumsum(clust_coeff_dist)
    clust_coef_dist = cs / node_nums
    return list(clust_coef_dist)


def calc_normalized_pdf_continuous(input_list, cc_interval = 0.02):
    len_list = len(input_list)
    list_sorted = sorted(input_list)
    list_dist = []
    #cc_list_len = int(1/cc_interval)

    start_interval = 0
    end_interval = start_interval + cc_interval
    count_interval = 0

    for cc in list_sorted:
        if cc <= end_interval:
            count_interval += 1
        else:
            list_dist.append(count_interval)
            end_interval += cc_interval
            count_interval = 0
            while cc > end_interval:
              list_dist.append(0)
              end_interval += cc_interval
            count_interval = 1


    list_dist.append(count_interval)
    #obtained_cc_list_len = len(clust_coeff_dist)
    #cc_list_diff = cc_list_len - obtained_cc_list_len

    #if obtained_cc_list_len < cc_list_len:
    #    clust_coeff_dist.extend([0] * cc_list_diff)

    list_dist = [a/len_list for a in list_dist]
    return list(list_dist)


def calc_normalized_cdf_continuous(input_list, cc_interval = 0.02):
    len_list = len(input_list)
    list_sorted = sorted(input_list)
    list_dist = []
    #cc_list_len = int(1/cc_interval)

    start_interval = 0
    end_interval = start_interval + cc_interval
    count_interval = 0

    for cc in list_sorted:
        if cc <= end_interval:
            count_interval += 1
        else:
            list_dist.append(count_interval)
            end_interval += cc_interval
            count_interval = 0
            while cc > end_interval:
              list_dist.append(0)
              end_interval += cc_interval
            count_interval = 1


    list_dist.append(count_interval)
    #obtained_cc_list_len = len(clust_coeff_dist)
    #cc_list_diff = cc_list_len - obtained_cc_list_len

    #if obtained_cc_list_len < cc_list_len:
    #    clust_coeff_dist.extend([0] * cc_list_diff)

    cs = np.cumsum(list_dist)
    list_dist = cs / len_list
    return list(list_dist)


def calc_weak_connected_components_sizes_dist(feature_extractor):
    print('calculating connected components sizes dist')
    feature_extractor.calc_connected_components()
    feature_extractor.calc_connected_components_sizes()
    # self.num_connected_components = cnx.number_connected_components(self.graph)
    return calc_cdf(feature_extractor.connected_components_sizes)


def calc_shortest_paths_CC_list(CC_snap):
    '''
        This function finds the shortest paths list in the given connected component snap graph.
        :param CC_snap:
        :return:
        '''
    shortest_path_lengths = []
    for node in CC_snap.Nodes():
        _, NIdToDistH = CC_snap.GetShortPathAll(node.GetId())
        for item in NIdToDistH:
            if item > node.GetId():
                shortest_path_lengths.append(NIdToDistH[item])

    print('shortest paths lengths ', shortest_path_lengths)
    return shortest_path_lengths


def calc_shortest_paths_CC_list_nx(CC):
    '''
        This function finds the shortest paths list in the given connected component snap graph.
        :param CC_snap:
        :return:
        '''
    shortest_path_lengths = []
    shortest_paths = dict(nx.shortest_path_length(CC))
    for node in shortest_paths:
        shortest_paths_of_node = shortest_paths[node]
        for item in shortest_paths_of_node:
            if item > node:
                shortest_path_lengths.append(shortest_paths_of_node[item])

    #print('shortest paths lengths ', shortest_path_lengths)
    return shortest_path_lengths


def calc_hop_plot_CC_dist_nx(CC):
    '''
    This function calculates hop-plots (or shortest path) distributions
     for the given connected components of a graph.
    :param CC_snap:
    :return:
    '''
    shortest_path_lengths = calc_shortest_paths_CC_list_nx(CC)
    if len(shortest_path_lengths) == 0:
        return [0]
    else:
        return calc_cdf(shortest_path_lengths)


def calc_hop_plot_dist_all_LCC_nx(graph_components):
    '''
    This function calculates hop-plots (or shortest path) distributions for all components of a graph and also
    returns hop-plots (or shortest path) distribution for the largest component.
    :param feature_extractor:
    :param graph_snap:
    :return:
    '''
    all_shortest_path_lengths = []
    LCC_shortest_path_lengths = []
    max_comp_size = 0
    for component in graph_components:
        component_size = len(component.nodes())
        component_shortest_paths_len = calc_shortest_paths_CC_list_nx(component)
        if component_size > max_comp_size:
            max_comp_size = component_size
            LCC_shortest_path_lengths = component_shortest_paths_len
        all_shortest_path_lengths.extend(component_shortest_paths_len)

    if len(all_shortest_path_lengths) == 0:
        return [0], [0]
    else:
        return calc_cdf(all_shortest_path_lengths), calc_cdf(LCC_shortest_path_lengths)


def calc_hop_plot_CC_dist(CC_snap):
    '''
    This function calculates hop-plots (or shortest path) distributions
     for the given connected components of a graph.
    :param CC_snap:
    :return:
    '''
    shortest_path_lengths = calc_shortest_paths_CC_list(CC_snap)
    if len(shortest_path_lengths) == 0:
        return [0]
    else:
        return calc_cdf(shortest_path_lengths)


def calc_hop_plot_dist(graph_components_snap):
    '''
    This function calculates hop-plots (or shortest path) distributions for all components of a graph.
    :param feature_extractor:
    :param graph_snap:
    :return:
    '''
    all_shortest_path_lengths = []
    for component in graph_components_snap:
        all_shortest_path_lengths.extend(calc_shortest_paths_CC_list(component))
    if len(all_shortest_path_lengths) == 0:
        return [0]
    else:
        return calc_cdf(all_shortest_path_lengths)


def calc_hop_plot_dist_nx(graph_components):
    '''
    This function calculates hop-plots (or shortest path) distributions for all components of a graph.
    :param feature_extractor:
    :param graph_snap:
    :return:
    '''
    all_shortest_path_lengths = []
    for component in graph_components:
        all_shortest_path_lengths.extend(calc_shortest_paths_CC_list_nx(component))

    if len(all_shortest_path_lengths) == 0:
        return [0]
    else:
        return calc_cdf(all_shortest_path_lengths)


def approx_sp_stat_distr_nk(graph_nk):
    '''
    Input: all graph component nodes
    Output: approx shortest path statistics & distributions (hop plot distri) for all and largest connected component ().
    '''
    all_shortest_path_counts = []
    all_hop_plots = []
    max_comp_size = 0
    indx_largest_comp = -1
    idx = 0
    cc = nk.components.ConnectedComponents(graph_nk)
    cc.run()
    print('number of comp ', cc.numberOfComponents())
    all_components = cc.getComponents()
    start_t = time.time()
    for comp in all_components:
      #print('comp is ', comp)
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
        
    # all components path features
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
    sp_info = {'shortest paths lengths min': min_shortest_path, 'shortest paths lengths max': max_shortest_path, 'shortest paths lengths mean': mean_shortest_path, 'shortest paths lengths var': var_shortest_path, 'shortest paths lengths distr': hop_plots_aggr_list}
    return sp_info, sp_lcc_info


def measure_num_Connected_Components_diff(graph, sampled_graph):
    original_graph_ncc = nx.number_connected_components(graph)
    sampled_graph_ncc = nx.number_connected_components(sampled_graph)
    return abs(original_graph_ncc - sampled_graph_ncc) / max(original_graph_ncc, sampled_graph_ncc)


def calc_KS_D_statistic(distribution_1, distribution_2):
    len_dist_1 = len(distribution_1)
    len_dist_2 = len(distribution_2)
    len_diff = abs(len_dist_1 - len_dist_2)
    if len_dist_1 < len_dist_2:
        distribution_1.extend([1]*len_diff)
    else:
        distribution_2.extend([1] * len_diff)

    norm_diff = abs(torch.tensor(distribution_1) - torch.tensor(distribution_2))
    #print('norm_diff ', norm_diff)
    return norm_diff.max()


def calc_Wasserstein_Distance(pdf_1, pdf_2):
    '''
    :param pdf_1: the first prob distr function (degree distribution)
    :param pdf_2: the second prob distr function (degree distribution)
    :return: Wasserstein distance
    '''
    len_dist_1 = len(pdf_1)
    len_dist_2 = len(pdf_2)
    len_diff = abs(len_dist_1 - len_dist_2)
    if len_dist_1 < len_dist_2:
        pdf_1.extend([0] * len_diff)
    else:
        pdf_2.extend([0] * len_diff)

    return wasserstein_distance(pdf_1, pdf_2)


def measure_preserved_property(graph, desired_property, sampled_graph, iter= 1):
    ###
    return measure_KS_D_statistics_graphs(graph, sampled_graph, desired_property)
    ...


def measure_Jaccard_Index(graph, sampled_graph):
    JI = 0
    for node in sampled_graph.nodes():
        node.ne



def eval_sampling_algorithm(graph, config, sampling_algorithm):

    ...

def eval_sampling_algorithms_batch(input_graphs, input_configs, sampling_algorithms):
    ...


if __name__ == "__main__":
    g1 = nx.barabasi_albert_graph(10, 2)
    #g2 = nx.barabasi_albert_graph(20, 2)
    #dd_1 = calc_normalized_degree_dist(g1, 10)
    #dd_2 = calc_normalized_degree_dist(g2, 20)
    '''
    print(g1.degree())
    print(dd_1)
    print(g2.degree())
    print(dd_2)
    print(calc_KS_D_statistic(dd_1, dd_2))
    print(nx.clustering(g1))
    print('starting calculation of normalized clust coeff distr')
    cc_dist_1 = calc_normalized_Clustering_Coeff_dist(g1, 10)
    print('clustering coef distr g1: ', cc_dist_1)
    #cc_dist_2 = calc_normalized_Clustering_Coeff_dist(g2, 20)
    #print('clustering coef distr g2: ', cc_dist_2)
    #print(calc_KS_D_statistic(cc_dist_1, cc_dist_2))
    #print(cc_dist)
    '''
    '''
    g = nx.erdos_renyi_graph(20, 0.05)
    nx.draw(g)
    plt.savefig('g')
    plt.show()
    gfe = graph_feature_extractor()
    gfe.set_graph(g)
    print('wcc dist ', calc_weak_connected_components_sizes_dist(gfe))
    gfe.calc_connected_components()
    gfe.calc_connected_components_sizes()
    largest_cc = max(gfe.connected_components_nodes, key=len)
    largest_cc_graph = gfe.graph.subgraph(largest_cc).copy()
    LCC_snap = gfe.convert_nx_to_snap_graph(largest_cc_graph)
    all_components_snap = []
    for components_nodes in gfe.connected_components_nodes:
        cc = gfe.graph.subgraph(components_nodes).copy()
        cc_snap = gfe.convert_nx_to_snap_graph(cc)
        all_components_snap.append(cc_snap)
    print('hop plot all ',calc_hop_plot_dist(all_components_snap))
    print('hop plot lcc ', calc_hop_plot_CC_dist(LCC_snap))
    '''
    
    root_folder = '/home/itec/haleh/PycharmProjects/Graph_Sampling_Alg_Selection_Small_Scale/venv/data/graphs_data/real_graphs/dataset_2/extracted_features/feature_set_1/temp/'
    '''
    input_graph_ids_pkl = ['ID_test_Real_Gowalla.csv.pickle']
    for graph_id in input_graph_ids_pkl:
        file_path = root_folder + graph_id + '_sp_LCC'
        in_dic = pickle.load(open(file_path, 'rb'))
        in_distr = calc_cdf(in_dic['shortest paths lengths'])
        out_dic = {'shortest paths lengths distr': in_distr, 'shortest_path_lengths_LCC_calc_time': in_dic['shortest_path_lengths_LCC_calc_time']}
        with open(root_folder + graph_id + 'sp_LCC_distr_info.json', 'w') as f:
            json.dump(out_dic, f)
   '''
    #input_graph_ids_for_json = ['ID_test_Real_tech-RL-caida.csv.pickle'] 
    input_graph_ids_for_json = ['ID_test_Real_ca-citeseer.csv.pickle']
    
    for graph_id in input_graph_ids_for_json:
        file_path = root_folder + graph_id + '_sp_LCC.json'
        in_dic = json.load(open(file_path, 'rb'))
        in_distr = calc_cdf(in_dic['shortest paths lengths'])
        out_dic = {'shortest paths lengths distr': in_distr, 'shortest_path_lengths_LCC_calc_time': in_dic['shortest_path_lengths_LCC_calc_time']}
        with open(root_folder + graph_id + 'sp_LCC_distr_info.json', 'w') as f:
            json.dump(out_dic, f)
    
     

