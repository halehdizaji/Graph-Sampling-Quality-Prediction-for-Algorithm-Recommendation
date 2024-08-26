###################################### This file is the same as Graph_Processing_fast_v1.py with added features ####################3
import time
import networkx as nx
import torch
import collections
from scipy.stats import entropy
import statistics
import numpy as np
#import cugraph as cnx
from multiprocessing import Pool
import itertools
import snap
import pickle


class graph_feature_extractor:
    def __init__(self):
        ...


    def set_graph(self, graph):
        self.graph = graph
        #self.initial_processing()


    def convert_nx_to_snap_graph(self, nx_graph):
        snap_graph = snap.TUNGraph.New()
        #snap_graph = snap.TUndirNet.New()
        for node in nx_graph.nodes():
            #print('node is ', int(node))
            snap_graph.AddNode(node)
            #print('node added.')

        for edge in nx.edges(nx_graph):
            snap_graph.AddEdge(edge[0], edge[1])

        return snap_graph


    def convert_snap_graph_to_nx(self, snap_graph):
        nx_graph = nx.Graph()
        nodes = []
        for node in snap_graph.Nodes():
            nodes.append(node.GetId())

        edges = []
        for edge in snap_graph.Edges():
            edges.append((edge.GetSrcNId(), edge.GetDstNId()))

        nx_graph.add_nodes_from(nodes)
        nx_graph.add_edges_from(edges)

        return nx_graph


    def initial_processing(self, feature_nums):
        print('starting graph init..')
        self.snap_graph = self.convert_nx_to_snap_graph(self.graph)
        self.feature_num = feature_nums
        self.calc_node_nums()

        print('calculating graph degrees')
        self.degrees = self.graph.degree()
        self.degrees_list = [self.degrees[i] for i in self.graph.nodes()]

        temp_start_time = time.time()
        print('calculating clustering coeff')
        start_time = time.time()
        ################################## Calculating Clustering Coefficient using networkx #########################
        #self.clust_coeff = nx.clustering(self.graph)
        ################################## Calculating Clustering Coefficient using snap ##########################
        self.clust_coeff = self.snap_graph.GetNodeClustCfAll()
        end_time = time.time()

        self.clust_coeff_calc_time = end_time - start_time
        self.clust_coeff_list = [self.clust_coeff[i] for i in self.graph.nodes()]

        print('calculating global clustering coefficient')
        start_time = time.time()
        self.calc_global_clust_coeff()
        #self.calc_global_clust_coeff_cnx()
        end_time = time.time()
        self.global_clust_coeff_calc_time = end_time - start_time

        print('calculating betweenness centrality')
        start_time = time.time()
        ##################### non-parallel version ######################
        #self.betweenness_centrality = cnx.betweenness_centrality(self.graph)
        ##################### Parallel version (Multiprocessing) ###########################
        '''
        self.betweenness_centrality = self.betweenness_centrality_parallel(self.graph)
        end_time = time.time()
        self.betweenness_centrality_calc_time = end_time - start_time
        nodes_in_betweenness_centrality = self.betweenness_centrality.keys()
        missing_nodes_betweenness_centrality = [i for i in self.graph.nodes if i not in nodes_in_betweenness_centrality]
        for i in missing_nodes_betweenness_centrality:
            self.betweenness_centrality[i] = 0
        self.betweenness_centrality_list = [self.betweenness_centrality[i] for i in self.graph.nodes()]
        '''
        ###################################################################

        ######################### Node & Edge Betweenness Centrality using Snap ########################
        nodes_betweenness, edges_betweenness = self.snap_graph.GetBetweennessCentr()
        end_time = time.time()
        self.node_edge_betweenness_centrality_calc_time = end_time - start_time
        self.node_betweenness_centrality_list = [nodes_betweenness[i] for i in nodes_betweenness]
        self.edge_betweenness_centrality_list = [edges_betweenness[i] for i in edges_betweenness]
        #print('node betweennesses ', self.node_betweenness_centrality_list)
        #print('edge betweennesses ', self.edge_betweenness_centrality_list)
        
        ######################## Hub & Authority Calculations using snap #########################

        ###############################

        ############################# Connected Components ########################
        print('calculating connected components')
        start_time = time.time()
        self.calc_connected_components()
        end_time = time.time()
        self.connected_components_calc_time = end_time - start_time
        self.number_connected_components = self.num_connected_components()
        self.calc_connected_components_sizes()
        #self.num_connected_components = cnx.number_connected_components(self.graph)
        temp_end_time = time.time()
        print('running of non-threaded functions is ', temp_end_time - temp_start_time)
        ############################# Eigenvector

        print('calculating eigenvector centrality ')
        start_time = time.time()
        ######### cugraph
        #self.eigenvector_centrality = cnx.eigenvector_centrality(self.graph)
        ######## nx
        self.eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=300, tol=1.0e-3)
        end_time = time.time()
        self.eigenvector_centrality_calc_time = end_time -start_time
        self.eigenvector_centrality_list = [self.eigenvector_centrality[i] for i in self.graph.nodes()]

        ############################# Pagerank centrality

        print('calculating pagerank centrality ')
        ######################################  Calculating PageRank Centrality using Cugraph #########################
        '''
        start_time = time.time()
        self.pagerank_centrality = cnx.pagerank(self.graph)
        end_time = time.time()
        self.pagerank_centrality_calc_time = end_time - start_time
        nodes_in_pagerank_centrality = self.pagerank_centrality.keys()
        missing_nodes_pagerank_centrality = [i for i in self.graph.nodes if i not in nodes_in_pagerank_centrality]
        for i in missing_nodes_pagerank_centrality:
            self.pagerank_centrality[i] = 0
        '''
        ##################################### Calculating PageRank Centrality using Snap ################################
        start_time = time.time()
        self.pagerank_centrality = self.snap_graph.GetPageRank()
        end_time = time.time()
        self.pagerank_centrality_calc_time = end_time - start_time

        ######
        self.pagerank_centrality_list = [self.pagerank_centrality[i] for i in self.graph.nodes()]

        ################################ calculating diameter of the graph ############
        '''
        print('calculating diameter of the graph ')
        start_time = time.time()
        self.diameter = nx.diameter(self.graph)
        end_time = time.time()
        self.diameter_calc_time = end_time - start_time
        '''

        ######################## Eccentricity Centrality of Nodes using snap ##############################
        largest_cc_graph_snap = self.convert_nx_to_snap_graph(largest_cc_graph)
        self.eccentricity_centrality_LCC_list = []
        print('calculating eccentricity centrality')
        start_time = time.time()
        for node in largest_cc_graph_snap.Nodes():
            self.eccentricity_centrality_LCC_list.append(largest_cc_graph_snap.GetNodeEcc(node.GetId(), True))
        end_time = time.time()
        #print('eccenticities: ', self.eccentricity_centrality_LCC_list)
        self.eccentricity_centrality_LCC_calc_time = end_time - start_time


        ######################## Farness Centrality of Nodes for the largest connected component using snap ##############################

        self.farness_centrality_list = []
        print('calculating farness centrality')
        start_time = time.time()
        for node in largest_cc_graph_snap.Nodes():
            self.farness_centrality_list.append(largest_cc_graph_snap.GetFarnessCentr(node.GetId(), True))
        end_time = time.time()
        self.farness_centrality_calc_time = end_time - start_time


        ############################# Shortest Paths ##############

        self.shortest_path_lengths = []
        start_time = time.time()
        for node in largest_cc_graph_snap.Nodes():
            _, NIdToDistH = largest_cc_graph_snap.GetShortPathAll(node.GetId())
            #print('number of node ', len(NIdToDistH.keys()))
            #print(NIdToDistH)
            for item in NIdToDistH:
                if item >= node.GetId():
                    self.shortest_path_lengths.append(NIdToDistH[item])
                    #nodes_pairs.append((node.GetId(), item))
                    #print(item, NIdToDistH[item])
        end_time = time.time()
        self.shortest_path_lengths_LCC_calc_time = end_time - start_time


        ############################### Degree Assortativity ############################

        start_time = time.time()
        self.degree_assortativity = nx.degree_assortativity_coefficient(self.graph)
        end_time = time.time()
        self.degree_assortativity_calc_time = end_time - start_time

        ################################ calculating max spanning tree #####################

        print('calculating max spanning tree (or min spanning tree)')
        start_time = time.time()
        self.calc_max_spanning_tree()
        end_time = time.time()
        self.max_spanning_tree_calc_time = end_time - start_time
        self.calc_degrees_max_spanning_tree()

        ############################### calculating maximal cliques ######################
        '''
        print('calculating maximal cliques..')
        start_time = time.time()
        self.find_maximal_cliques()
        end_time = time.time()
        self.maximal_cliques_calc_time = end_time - start_time
        self.calc_maximal_cliques_sizes()
        '''

        ############################## Calculating communities using snap #########################
        '''
        print('calculating communities..')
        print('node nums: ', self.snap_graph.GetNodes())
        print('')
        start_time = time.time()
        self.modularity, CmtyV = self.snap_graph.CommunityCNM()
        end_time = time.time()
        self.communities = []
        self.num_communities = 0
        self.communities_sizes = []
        for Cmty in CmtyV:
            community = []
            self.num_communities += 1
            for NI in Cmty:
                community.append(NI)
            self.communities_sizes.append(len(community))
            self.communities.append(community)
        #print('communities sizes: ', self.communities_sizes)
        self.communities_calc_time = end_time - start_time
        '''

        ######################################################

        self.calc_edge_nums()
        self.calc_normalized_degree_dist()
        self.calc_normalized_clust_coeff_dist()

        print('graph init finished.')



    def initial_processing_read_features(self):
        '''
        This function initializes the object, using saved extracted information of the graph with graph_file_path (without path features)
        :param graph_id:
        :return:
        '''
        print('starting graph init..')
        self.snap_graph = self.convert_nx_to_snap_graph(self.graph)
        self.calc_node_nums()

        print('calculating graph degrees')
        self.degrees = self.graph.degree()
        self.degrees_list = [self.degrees[i] for i in self.graph.nodes()]

        print('reading clustering coeff')
        ################################## Calculating Clustering Coefficient using networkx #########################
        # self.clust_coeff = nx.clustering(self.graph)
        ################################## Calculating Clustering Coefficient using snap ##########################
        self.read_clustering_coeff_info()

        print('reading global clustering coefficient')
        self.read_global_clustering_info()

        print('reading betweenness centrality')
        self.read_betweenness_info()

        ############################# Connected Components ########################
        print('reading connected components info only')
        self.read_connected_components_info_only()

        print('reading eigenvector centrality ')
        self.read_eigenvector_info()

        ############################# Pagerank centrality

        print('reading pagerank centrality ')
        self.read_pagerank_info()


        ############################### calculating diameter of the largest component #####
        #print('reading diameter')
        #self.diameter_largest_cc = max(self.shortest_path_lengths)


        # Eccentricity
        self.read_eccentricity_LCC_info()

        ############################### Degree Assortativity ############################
        self.read_assortativity_info()

        ################################ max spanning tree #####################

        print('reading max spanning tree (or min spanning tree)')
        self.read_max_spanning_tree_info()

        self.calc_edge_nums()
        self.calc_normalized_degree_dist()
        #to be corrected
        self.calc_normalized_clust_coeff_dist()

        print('graph init from saved data finished.')


    def set_extracted_features_folder(self, folder_path):
        self.features_folder = folder_path


    def set_graph_id(self, graph_id):
        self.graph_id = graph_id


    def read_clustering_coeff_info(self):
        clus_coeff_info = pickle.load(open(self.features_folder + 'ID_' + self.graph_id + '.pickle_clustering_coeff', 'rb'))
        self.clust_coeff_list = clus_coeff_info['clust coeff']
        self.clust_coeff_calc_time = clus_coeff_info['calc time']


    def read_assortativity_info(self):
        assortativity_info = pickle.load(
            open(self.features_folder + 'ID_' + self.graph_id + '.pickle_assortativity', 'rb'))
        self.degree_assortativity = assortativity_info['degree_assortativity']
        self.degree_assortativity_calc_time = assortativity_info['degree_assortativity_calc_time']


    def read_betweenness_info(self):
        betweenness_info = pickle.load(
            open(self.features_folder + 'ID_' + self.graph_id + '.pickle_betweenness', 'rb'))
        self.edge_betweenness_centrality_list = betweenness_info['edge_betweenness_centrality_list']
        self.node_betweenness_centrality_list = betweenness_info['node_betweenness_centrality_list']
        self.node_edge_betweenness_centrality_calc_time = betweenness_info['calc time']


    def read_max_spanning_tree_info(self):
        mst_info = pickle.load(
            open(self.features_folder + 'ID_' + self.graph_id + '.pickle_mst_deg', 'rb'))
        self.max_spanning_tree_degrees = mst_info['MST degrees']
        self.max_spanning_tree_calc_time = mst_info['calc time']


    def read_global_clustering_info(self):
        global_clus_coeff_info = pickle.load(
            open(self.features_folder + 'ID_' + self.graph_id + '.pickle_global_clustering_coeff', 'rb'))
        self.global_clust_coeff = global_clus_coeff_info['global clust coeff']
        self.global_clust_coeff_calc_time = global_clus_coeff_info['calc time']


    def read_eigenvector_info(self):
        eigenvector_info = pickle.load(
            open(self.features_folder + 'ID_' + self.graph_id + '.pickle_eigenvector', 'rb'))
        self.eigenvector_centrality_list = eigenvector_info['EV']
        self.eigenvector_centrality_calc_time = eigenvector_info['calc time']


    def read_pagerank_info(self):
        pagerank_info = pickle.load(
            open(self.features_folder + 'ID_' + self.graph_id + '.pickle_pagerank', 'rb'))
        self.pagerank_centrality_list = pagerank_info['PR']
        self.pagerank_centrality_calc_time = pagerank_info['calc time']


    def read_connected_components_info(self):
        connected_components_info = pickle.load(
            open(self.features_folder + 'ID_' + self.graph_id + '.pickle_connected_comp', 'rb'))
        self.number_connected_components = connected_components_info['num cc']
        self.connected_components_sizes = connected_components_info['cc sizes']
        self.eccentricity_centrality_LCC_list = connected_components_info['ecc list']
        self.shortest_path_lengths = connected_components_info['shortest paths lengths']
        self.shortest_path_lengths_LCC_calc_time = connected_components_info['shortest_path_lengths_LCC_calc_time']
        self.eccentricity_centrality_LCC_calc_time = connected_components_info['ecc calc time']
        self.connected_components_calc_time = connected_components_info['cc calc time']
        self.farness_centrality_list = connected_components_info['farness']
        self.farness_centrality_calc_time = connected_components_info['farness calc time']

    def read_connected_components_info_only(self):
        connected_components_info = pickle.load(
            open(self.features_folder + 'ID_' + self.graph_id + '.pickle_connected_comp', 'rb'))
        self.number_connected_components = connected_components_info['num cc']
        self.connected_components_sizes = connected_components_info['cc sizes']
        self.connected_components_calc_time = connected_components_info['cc calc time']

    def read_eccentricity_LCC_info(self):
        connected_components_info = pickle.load(
            open(self.features_folder + 'ID_' + self.graph_id + '.pickle_ecc_LCC', 'rb'))
        self.eccentricity_centrality_LCC_list = connected_components_info['ecc list']
        self.eccentricity_centrality_LCC_calc_time = connected_components_info['ecc calc time']

    def calc_node_nums(self):
        self.node_nums = len(self.graph.nodes())


    def calc_edge_nums(self):
        self.edge_nums = len(self.graph.edges())


    def min_degree(self):
        return min(self.degrees_list)


    def max_degree(self):
        return max(self.degrees_list)


    def mean_degree(self):
        return sum(self.degrees_list)/self.node_nums


    def var_degree(self):
        return np.var(self.degrees_list)


    def median_degree(self):
        return np.median(self.degrees_list)


    def graph_density(self):
        return 2 * len(self.graph.edges())/(self.node_nums * (self.node_nums - 1))


    def min_eccentricity(self):
        return min(self.eccentricity_centrality_LCC_list)


    def max_eccentricity(self):
        return max(self.eccentricity_centrality_LCC_list)


    def mean_eccentricity(self):
        return np.mean(self.eccentricity_centrality_LCC_list)


    def median_eccentricity(self):
        return np.median(self.eccentricity_centrality_LCC_list)


    def var_eccentricity(self):
        return np.var(self.eccentricity_centrality_LCC_list)


    def min_community_size(self):
        return min(self.communities_sizes)


    def max_community_size(self):
        return max(self.communities_sizes)


    def mean_community_size(self):
        return np.mean(self.communities_sizes)


    def median_community_size(self):
        return np.median(self.communities_sizes)


    def var_community_size(self):
        return np.var(self.communities_sizes)


    def min_farness_centrality(self):
        return min(self.farness_centrality_list)


    def max_farness_centrality(self):
        return max(self.farness_centrality_list)


    def mean_farness_centrality(self):
        return np.mean(self.farness_centrality_list)


    def median_farness_centrality(self):
        return np.median(self.farness_centrality_list)


    def var_farness_centrality(self):
        return np.var(self.farness_centrality_list)


    def min_shortest_path_length_LCC(self):
        return np.min(self.shortest_path_lengths)


    def max_shortest_path_length_LCC(self):
        return np.max(self.shortest_path_lengths)


    def mean_shortest_path_length_LCC(self):
        return np.mean(self.shortest_path_lengths)


    def var_shortest_path_length_LCC(self):
        return np.var(self.shortest_path_lengths)


    def median_shortest_path_length_LCC(self):
        return np.median(self.shortest_path_lengths)


    def min_edge_betweenness_centrality(self):
        return min(self.edge_betweenness_centrality_list)


    def max_edge_betweenness_centrality(self):
        return max(self.edge_betweenness_centrality_list)


    def mean_edge_betweenness_centrality(self):
        return sum(self.edge_betweenness_centrality_list) / self.node_nums


    def var_edge_betweenness_centrality(self):
        return np.var(self.edge_betweenness_centrality_list)


    def median_edge_betweenness_centrality(self):
        return np.median(self.edge_betweenness_centrality_list)


    def calc_clust_coeff_snap(self):
        return snap.GetNodeClustCfAll(self.snap_graph)


    def calc_global_clust_coeff(self):
        num_triples = sum([d*(d -1)/2 for d in self.degrees_list])
        if num_triples == 0:
            self.global_clust_coeff = 0
        else:
            self.global_clust_coeff = sum(nx.triangles(self.graph).values())/ num_triples


    def calc_global_clust_coeff_cnx(self):
        num_triples = sum([d*(d -1)/2 for d in self.degrees_list])
        if num_triples == 0:
            self.global_clust_coeff = 0
        else:
            self.global_clust_coeff = cnx.triangle_count(self.graph).sum()[1] / num_triples


    def min_clust_coeff(self):
        return min(self.clust_coeff_list)


    def max_clust_coeff(self):
        return max(self.clust_coeff_list)


    def mean_clust_coeff(self):
        return sum(self.clust_coeff_list) / self.node_nums


    def var_clust_coeff(self):
        return np.var(self.clust_coeff_list)


    def median_clust_coeff(self):
        return np.median(self.clust_coeff_list)


    def num_connected_components(self):
        #return self.number_connected_components
        return len(self.connected_components_nodes)


    def calc_connected_components(self):
        self.connected_components = nx.connected_components(self.graph)
        #print('connected components ', [c for c in self.connected_components])
        self.connected_components_nodes = [comp_nodes for comp_nodes in self.connected_components]
        #print('connected components nodes ', self.connected_components_nodes)


    def diameter(self):
        return nx.diameter(self.graph)


    def diameter_largest_component(self):

        return diameter_largest_cc


    def diameter_component(self, component):
        return nx.diameter(component)


    def min_node_betweenness_centrality(self):
        return min(self.node_betweenness_centrality_list)


    def max_node_betweenness_centrality(self):
        return max(self.node_betweenness_centrality_list)


    def mean_node_betweenness_centrality(self):
        return sum(self.node_betweenness_centrality_list) / self.node_nums


    def var_node_betweenness_centrality(self):
        return np.var(self.node_betweenness_centrality_list)


    def median_node_betweenness_centrality(self):
        return np.median(self.node_betweenness_centrality_list)


    def min_eigenvector_centrality(self):
        return min(self.eigenvector_centrality_list)


    def max_eigenvector_centrality(self):
        return max(self.eigenvector_centrality_list)


    def mean_eigenvector_centrality(self):
        return sum(self.eigenvector_centrality_list) / self.node_nums


    def var_eigenvector_centrality(self):
        return np.var(self.eigenvector_centrality_list)


    def median_eigenvector_centrality(self):
        return np.median(self.eigenvector_centrality_list)


    def min_pagerank_centrality(self):
        return min(self.pagerank_centrality_list)


    def max_pagerank_centrality(self):
        return max(self.pagerank_centrality_list)


    def mean_pagerank_centrality(self):
        return sum(self.pagerank_centrality_list) / self.node_nums


    def var_pagerank_centrality(self):
        return np.var(self.pagerank_centrality_list)


    def median_pagerank_centrality(self):
        return np.median(self.pagerank_centrality_list)


    def calc_normalized_degree_dist(self):
        degreeCount = collections.Counter(sorted(self.degrees_list))
        deg, cnt = zip(*degreeCount.items())
        deg_list = list(deg)
        cnt_list = list(cnt)
        cnt_all = [0] * (self.node_nums)
        for i in range(len(deg_list)):
            cnt_all[deg_list[i]] = cnt_list[i]/self.node_nums
        self.degree_dist = cnt_all


    def calc_normalized_clust_coeff_dist(self,cc_interval=0.02):
        clust_coeff = nx.clustering(self.graph)
        print('finished calculating clustering coefficients')
        clust_coeff_sorted = sorted(clust_coeff.values())
        self.clust_coeff_dist = []
        # cc_list_len = int(1/cc_interval)

        start_interval = 0
        end_interval = start_interval + cc_interval
        count_interval = 0

        for cc in clust_coeff_sorted:
            if cc <= end_interval:
                count_interval += 1
            else:
                self.clust_coeff_dist.append(count_interval)
                count_interval = 1
                end_interval += cc_interval

        self.clust_coeff_dist.append(count_interval)
        # obtained_cc_list_len = len(clust_coeff_dist)
        # cc_list_diff = cc_list_len - obtained_cc_list_len

        # if obtained_cc_list_len < cc_list_len:
        #    clust_coeff_dist.extend([0] * cc_list_diff)

        self.normal_clust_coeff_dist = [d/self.node_nums for d in self.clust_coeff_dist]


    def calc_max_spanning_tree(self):
        self.max_spanning_tree = nx.maximum_spanning_tree(self.graph)


    def calc_dominating_set(self):
        self.dominating_set = nx.dominating_set(self.graph)


    def calc_connected_components_sizes(self):
        self.connected_components_sizes = [len(c) for c in self.connected_components_nodes]


    def calc_min_connected_component_size(self):
        return min(self.connected_components_sizes)


    def calc_max_connected_component_size(self):
        return max(self.connected_components_sizes)


    def calc_mean_connected_component_size(self):
        return np.mean(self.connected_components_sizes)


    def calc_median_connected_component_size(self):
        return np.median(self.connected_components_sizes)


    def calc_var_connected_component_size(self):
        return np.var(self.connected_components_sizes)


    def calc_degrees_max_spanning_tree(self):
        self.max_spanning_tree_degrees = [self.max_spanning_tree.degree()[i] for i in self.max_spanning_tree.nodes()]


    def calc_min_degrees_max_spanning_tree(self):
        return min(self.max_spanning_tree_degrees)


    def calc_max_degrees_max_spanning_tree(self):
        return max(self.max_spanning_tree_degrees)


    def calc_mean_degrees_max_spanning_tree(self):
        return np.mean(self.max_spanning_tree_degrees)


    def calc_median_degrees_max_spanning_tree(self):
        return np.median(self.max_spanning_tree_degrees)


    def calc_var_degrees_max_spanning_tree(self):
        return np.var(self.max_spanning_tree_degrees)


    def find_maximal_cliques(self):
        self.maximal_cliques_nodes = [clique for clique in nx.find_cliques(self.graph)]


    def calc_maximal_cliques_sizes(self):
        self.maximal_cliques_sizes = [len(clique) for clique in self.maximal_cliques_nodes]


    def calc_number_maximal_cliques(self):
        return len(self.maximal_cliques_nodes)


    def calc_min_maximal_clique_size(self):
        return min(self.maximal_cliques_sizes)


    def calc_max_maximal_clique_size(self):
        return max(self.maximal_cliques_sizes)


    def calc_mean_maximal_clique_size(self):
        return np.mean(self.maximal_cliques_sizes)


    def calc_median_maximal_clique_size(self):
        return np.median(self.maximal_cliques_sizes)


    def calc_var_maximal_clique_size(self):
        return np.var(self.maximal_cliques_sizes)


    def chunks(self, l, n):
        """Divide a list of nodes `l` in `n` chunks"""
        l_c = iter(l)
        while 1:
            x = tuple(itertools.islice(l_c, n))
            if not x:
                return
            yield x


    def node_betweenness_centrality_parallel(self, G, processes=None):
        """Parallel betweenness centrality  function"""
        p = Pool(processes=processes)
        node_divisor = len(p._pool) * 4
        node_chunks = list(self.chunks(G.nodes(), G.order() // node_divisor))
        num_chunks = len(node_chunks)
        bt_sc = p.starmap(
            nx.betweenness_centrality_subset,
            zip(
                [G] * num_chunks,
                node_chunks,
                [list(G)] * num_chunks,
                [True] * num_chunks,
                [None] * num_chunks,
            ),
        )

        # Reduce the partial solutions
        bt_c = bt_sc[0]
        for bt in bt_sc[1:]:
            for n in bt:
                bt_c[n] += bt[n]
        return bt_c


    def calc_graph_features_normal(self):
        self.graph_features_dic = {}
        self.graph_features_dic['node_nums'] = (self.node_nums - self.min_node_nums) / (self.max_node_nums - self.min_node_nums)
        self.graph_features_dic['edge_nums'] = (self.edge_nums - self.min_edge_nums) / (self.max_edge_nums - self.min_edge_nums)
        self.graph_features_dic['min_degree'] = (self.min_degree() - self.min_min_degree) / (self.max_min_degree - self.min_min_degree)
        self.graph_features_dic['max_degree'] = (self.max_degree() - self.min_max_degree) / (self.max_max_degree - self.min_max_degree)
        self.graph_features_dic['mean_degree'] = (self.mean_degree() - self.min_mean_degree) / (self.max_mean_degree - self.min_mean_degree)
        self.graph_features_dic['var_degree'] = (self.var_degree() - self.min_var_degree) / (self.max_var_degree - self.min_var_degree)
        self.graph_features_dic['median_degree'] = (self.median_degree() - self.min_median_degree) / (
                    self.max_median_degree - self.min_median_degree)
        self.graph_features_dic['graph_density'] = (self.graph_density() - self.min_graph_density) / (
                    self.max_graph_density - self.min_graph_density)
        self.graph_features_dic['min_clust_coeff'] = (self.min_clust_coeff() - self.min_min_clust_coeff) / (
                    self.max_min_clust_coeff - self.min_min_clust_coeff)
        self.graph_features_dic['max_clust_coeff'] = (self.max_clust_coeff() - self.min_max_clust_coeff) / (
                    self.max_max_clust_coeff - self.min_max_clust_coeff)
        self.graph_features_dic['mean_clust_coeff'] = (self.mean_clust_coeff() - self.min_mean_clust_coeff) / (
                    self.max_mean_clust_coeff - self.min_mean_clust_coeff)
        self.graph_features_dic['var_clust_coeff'] = (self.var_clust_coeff() - self.min_var_clust_coeff) / (
                    self.max_var_clust_coeff - self.min_var_clust_coeff)
        self.graph_features_dic['median_clust_coeff'] = (self.median_clust_coeff() - self.min_median_clust_coeff) / (
                self.max_median_clust_coeff - self.min_median_clust_coeff)
        self.graph_features_dic['clust_coeff_calc_time'] = (self.clust_coeff_calc_time - self.min_clust_coeff_calc_time) / (
                    self.max_clust_coeff_calc_time - self.min_clust_coeff_calc_time)
        self.graph_features_dic['min_node_betweenness_centrality'] = (self.min_node_betweenness_centrality() - self.min_min_node_betweenness_centrality) / (
                    self.max_min_node_betweenness_centrality - self.min_min_node_betweenness_centrality)
        self.graph_features_dic['max_node_betweenness_centrality'] = (self.max_node_betweenness_centrality() - self.min_max_node_betweenness_centrality) / (
                    self.max_max_node_betweenness_centrality - self.min_max_node_betweenness_centrality)
        self.graph_features_dic['mean_node_betweenness_centrality'] = (self.mean_node_betweenness_centrality() - self.min_mean_node_betweenness_centrality) / (
                    self.max_mean_node_betweenness_centrality - self.min_mean_node_betweenness_centrality)
        self.graph_features_dic['var_node_betweenness_centrality'] = (self.var_node_betweenness_centrality() - self.min_var_node_betweenness_centrality) / (
                self.max_var_node_betweenness_centrality - self.min_var_node_betweenness_centrality)
        self.graph_features_dic['median_node_betweenness_centrality'] = (self.median_node_betweenness_centrality() - self.min_median_node_betweenness_centrality) / (
                self.max_median_node_betweenness_centrality - self.min_median_node_betweenness_centrality)
        self.graph_features_dic['node_edge_betweenness_centrality_calc_time'] = (self.node_edge_betweenness_centrality_calc_time - self.min_node_edge_betweenness_centrality_calc_time) / (
                    self.max_node_edge_betweenness_centrality_calc_time - self.min_node_edge_betweenness_centrality_calc_time)
        self.graph_features_dic['min_edge_betweenness_centrality'] = (
                                                                                 self.min_edge_betweenness_centrality() - self.min_min_edge_betweenness_centrality) / (
                                                                             self.max_min_edge_betweenness_centrality - self.min_min_edge_betweenness_centrality)
        self.graph_features_dic['max_edge_betweenness_centrality'] = (
                                                                                 self.max_edge_betweenness_centrality() - self.min_max_edge_betweenness_centrality) / (
                                                                             self.max_max_edge_betweenness_centrality - self.min_max_edge_betweenness_centrality)
        self.graph_features_dic['mean_edge_betweenness_centrality'] = (
                                                                                  self.mean_edge_betweenness_centrality() - self.min_mean_edge_betweenness_centrality) / (
                                                                              self.max_mean_edge_betweenness_centrality - self.min_mean_edge_betweenness_centrality)
        self.graph_features_dic['var_edge_betweenness_centrality'] = (
                                                                                 self.var_edge_betweenness_centrality() - self.min_var_edge_betweenness_centrality) / (
                                                                             self.max_var_edge_betweenness_centrality - self.min_var_edge_betweenness_centrality)
        self.graph_features_dic['median_edge_betweenness_centrality'] = (self.median_edge_betweenness_centrality() - self.min_median_edge_betweenness_centrality) / (
                                                                                self.max_median_edge_betweenness_centrality - self.min_median_edge_betweenness_centrality)
        #self.graph_features_dic['edge_betweenness_centrality_calc_time'] = (self.edge_betweenness_centrality_calc_time - self.min_edge_betweenness_centrality_calc_time) / (
        #                                                                           self.max_edge_betweenness_centrality_calc_time - self.min_edge_betweenness_centrality_calc_time)
        #self.graph_features_dic['min_farness_centrality'] = (self.min_farness_centrality() - self.min_min_farness_centrality) / (self.max_min_farness_centrality - self.min_min_farness_centrality)
        #self.graph_features_dic['max_farness_centrality'] = (self.max_farness_centrality() - self.min_max_farness_centrality) / (
        #                                                                self.max_max_farness_centrality - self.min_max_farness_centrality)
        #self.graph_features_dic['mean_farness_centrality'] = (self.mean_farness_centrality() - self.min_mean_farness_centrality) / (
        #                                                            self.max_mean_farness_centrality - self.min_mean_farness_centrality)
        #self.graph_features_dic['var_farness_centrality'] = (self.var_farness_centrality() - self.min_var_farness_centrality) / (
        #                                                             self.max_var_farness_centrality - self.min_var_farness_centrality)
        #self.graph_features_dic['median_farness_centrality'] = (self.median_farness_centrality() - self.min_median_farness_centrality) / (
        #                                                            self.max_median_farness_centrality - self.min_median_farness_centrality)
        #self.graph_features_dic['min_community_size'] = (self.min_community_size() - self.min_min_community_size)/(self.max_min_community_size - self.min_min_community_size)
        #self.graph_features_dic['max_community_size'] = (self.max_community_size() - self.min_max_community_size) / (
        #            self.max_max_community_size - self.min_max_community_size)
        #self.graph_features_dic['mean_community_size'] = (self.mean_community_size() - self.min_mean_community_size) / (
        #        self.max_mean_community_size - self.min_mean_community_size)
        #self.graph_features_dic['var_community_size'] = (self.var_community_size() - self.min_var_community_size) / (
        #                                                         self.max_var_community_size - self.min_var_community_size)
        #self.graph_features_dic['median_community_size'] = (self.median_community_size() - self.min_median_community_size) / (
        #                                                         self.max_median_community_size - self.min_median_community_size)
        #self.graph_features_dic['num_communities'] = (self.num_communities - self.min_num_communities)/(self.max_num_communities - self.min_num_communities)
        #self.graph_features_dic['modularity'] = (self.modularity - self.min_modularity) / (self.max_modularity - self.min_modularity)
        self.graph_features_dic['min_eccentricity_centrality'] = (self.min_eccentricity() - self.min_min_eccentricity) / (self.max_min_eccentricity - self.min_min_eccentricity)
        self.graph_features_dic['max_eccentricity_centrality'] = (self.max_eccentricity() - self.min_max_eccentricity) / (
                                                                             self.max_max_eccentricity - self.min_max_eccentricity)
        self.graph_features_dic['mean_eccentricity_centrality'] = (self.mean_eccentricity() - self.min_mean_eccentricity) / (
                                                                         self.max_mean_eccentricity - self.min_mean_eccentricity)
        self.graph_features_dic['median_eccentricity_centrality'] = (self.median_eccentricity() - self.min_median_eccentricity) / (
                                                                          self.max_median_eccentricity - self.min_median_eccentricity)
        self.graph_features_dic['var_eccentricity_centrality'] = (self.var_eccentricity() - self.min_var_eccentricity) / (
                                                                          self.max_var_eccentricity - self.min_var_eccentricity)
        self.graph_features_dic['num_connected_components'] = (self.num_connected_components() - self.min_num_connected_components) / (
                    self.max_num_connected_components - self.min_num_connected_components)
        self.graph_features_dic['connected_components_calc_time'] = (self.connected_components_calc_time - self.min_connected_components_calc_time) / (
                    self.max_connected_components_calc_time - self.min_connected_components_calc_time)
        self.graph_features_dic['node_nums_div_edge_nums'] = (self.node_nums / self.edge_nums - self.min_node_nums_div_edge_nums) / (
                    self.max_node_nums_div_edge_nums - self.min_node_nums_div_edge_nums)
        self.graph_features_dic['edge_nums_div_node_nums'] = (self.edge_nums / self.node_nums - self.min_edge_nums_div_node_nums) / (
                    self.max_edge_nums_div_node_nums - self.min_edge_nums_div_node_nums)
        self.graph_features_dic['entropy_degrees'] = entropy(self.degree_dist)
        self.graph_features_dic['entropy_clust_coeff'] = entropy(self.normal_clust_coeff_dist)
        self.graph_features_dic['min_eigenvector_centrality'] = self.min_eigenvector_centrality()
        self.graph_features_dic['max_eigenvector_centrality'] = self.max_eigenvector_centrality()
        self.graph_features_dic['mean_eigenvector_centrality'] = self.mean_eigenvector_centrality()
        self.graph_features_dic['var_eigenvector_centrality'] = self.var_eigenvector_centrality()
        self.graph_features_dic['eigenvector_centrality_calc_time'] = self.eigenvector_centrality_calc_time
        self.graph_features_dic['min_pagerank_centrality'] = (self.min_pagerank_centrality() - self.min_min_pagerank_centrality) / (
                    self.max_min_pagerank_centrality - self.min_min_pagerank_centrality)
        self.graph_features_dic['max_pagerank_centrality'] = (self.max_pagerank_centrality() - self.min_max_pagerank_centrality) / (
                    self.max_max_pagerank_centrality - self.min_max_pagerank_centrality)
        self.graph_features_dic['mean_pagerank_centrality'] = (self.mean_pagerank_centrality() - self.min_mean_pagerank_centrality) / (
                    self.max_mean_pagerank_centrality - self.min_mean_pagerank_centrality)
        self.graph_features_dic['var_pagerank_centrality'] = (self.var_pagerank_centrality() - self.min_var_pagerank_centrality) / (
                self.max_var_pagerank_centrality - self.min_var_pagerank_centrality)
        self.graph_features_dic['median_pagerank_centrality'] = (self.median_pagerank_centrality() - self.min_median_pagerank_centrality) / (
                self.max_median_pagerank_centrality - self.min_median_pagerank_centrality)
        self.graph_features_dic['pagerank_centrality_calc_time'] = (self.pagerank_centrality_calc_time - self.min_pagerank_centrality_calc_time) / (
                    self.max_pagerank_centrality_calc_time - self.min_pagerank_centrality_calc_time)
        self.graph_features_dic['max_degrees_max_spanning_tree'] = (self.calc_max_degrees_max_spanning_tree() - self.min_max_degree_spanning_tree) /  (self.max_max_degree_spanning_tree - self.min_max_degree_spanning_tree)
        self.graph_features_dic['min_degrees_max_spanning_tree'] = (self.calc_min_degrees_max_spanning_tree() - self.min_min_degree_spanning_tree) / (self.max_min_degree_spanning_tree - self.min_min_degree_spanning_tree)
        self.graph_features_dic['mean_degrees_max_spanning_tree'] = (self.calc_mean_degrees_max_spanning_tree() - self.min_mean_degree_spanning_tree) / (self.max_mean_degree_spanning_tree - self.min_mean_degree_spanning_tree)
        self.graph_features_dic['var_degrees_max_spanning_tree'] = (self.calc_var_degrees_max_spanning_tree() - self.min_var_degree_spanning_tree) / (self.max_var_degree_spanning_tree - self.min_var_degree_spanning_tree)
        self.graph_features_dic['max_spanning_tree_calc_time'] = (self.max_spanning_tree_calc_time - self.min_max_spanning_tree_calc_time) / (
                    self.max_max_spanning_tree_calc_time - self.min_max_spanning_tree_calc_time)
        self.graph_features_dic['diameter'] = (self.diameter_largest_cc - self.min_diameter_largest_component) / (self.max_diameter_largest_component - self.min_diameter_largest_component)
        self.graph_features_dic['diameter_calc_time'] = (self.diameter_calc_time - self.min_diameter_calc_time) / (self.max_diameter_calc_time - self.min_diameter_calc_time)
        self.graph_features_dic['global_clust_coeff'] = self.global_clust_coeff
        self.graph_features_dic['global_clust_coeff_calc_time'] = self.global_clust_coeff_calc_time


    def calc_graph_features_non_normal(self):
        self.graph_features_dic = {}
        self.graph_features_dic['node_nums'] = self.node_nums
        self.graph_features_dic['edge_nums'] = self.edge_nums
        self.graph_features_dic['min_degree'] = self.min_degree()
        self.graph_features_dic['max_degree'] = self.max_degree()
        self.graph_features_dic['mean_degree'] = self.mean_degree()
        self.graph_features_dic['var_degree'] = self.var_degree()
        self.graph_features_dic['median_degree'] = self.median_degree()
        self.graph_features_dic['graph_density'] = self.graph_density()
        self.graph_features_dic['min_clust_coeff'] = self.min_clust_coeff()
        self.graph_features_dic['max_clust_coeff'] = self.max_clust_coeff()
        self.graph_features_dic['mean_clust_coeff'] = self.mean_clust_coeff()
        self.graph_features_dic['var_clust_coeff'] = self.var_clust_coeff()
        self.graph_features_dic['median_clust_coeff'] = self.median_clust_coeff()
        self.graph_features_dic['clust_coeff_calc_time'] = self.clust_coeff_calc_time
        self.graph_features_dic['min_node_betweenness_centrality'] = self.min_node_betweenness_centrality()
        self.graph_features_dic['max_node_betweenness_centrality'] = self.max_node_betweenness_centrality()
        self.graph_features_dic['mean_node_betweenness_centrality'] = self.mean_node_betweenness_centrality()
        self.graph_features_dic['var_node_betweenness_centrality'] = self.var_node_betweenness_centrality()
        self.graph_features_dic['median_node_betweenness_centrality'] = self.median_node_betweenness_centrality()
        self.graph_features_dic['node_edge_betweenness_centrality_calc_time'] = self.node_edge_betweenness_centrality_calc_time
        self.graph_features_dic['min_edge_betweenness_centrality'] = self.min_edge_betweenness_centrality()
        self.graph_features_dic['max_edge_betweenness_centrality'] = self.max_edge_betweenness_centrality()
        self.graph_features_dic['mean_edge_betweenness_centrality'] = self.mean_edge_betweenness_centrality()
        self.graph_features_dic['var_edge_betweenness_centrality'] = self.var_edge_betweenness_centrality()
        self.graph_features_dic['median_edge_betweenness_centrality'] = self.median_edge_betweenness_centrality()
        self.graph_features_dic['min_eccentricity_centrality'] = self.min_eccentricity()
        self.graph_features_dic['max_eccentricity_centrality'] = self.max_eccentricity()
        self.graph_features_dic['mean_eccentricity_centrality'] = self.mean_eccentricity()
        self.graph_features_dic['median_eccentricity_centrality'] = self.median_eccentricity()
        self.graph_features_dic['var_eccentricity_centrality'] = self.var_eccentricity()
        self.graph_features_dic['num_connected_components'] = self.number_connected_components
        self.graph_features_dic['max_connected_components_size'] = self.calc_max_connected_component_size()
        self.graph_features_dic['min_connected_components_size'] = self.calc_min_connected_component_size()
        self.graph_features_dic['mean_connected_components_size'] = self.calc_mean_connected_component_size()
        self.graph_features_dic['var_connected_components_size'] = self.calc_var_connected_component_size()
        self.graph_features_dic['median_connected_components_size'] = self.calc_median_connected_component_size()
        self.graph_features_dic['connected_components_calc_time'] = self.connected_components_calc_time
        self.graph_features_dic['node_nums_div_edge_nums'] = self.node_nums / self.edge_nums
        self.graph_features_dic['edge_nums_div_node_nums'] = self.edge_nums / self.node_nums
        self.graph_features_dic['entropy_degrees'] = entropy(self.degree_dist)
        self.graph_features_dic['entropy_clust_coeff'] = entropy(self.normal_clust_coeff_dist)
        self.graph_features_dic['min_eigenvector_centrality'] = self.min_eigenvector_centrality()
        self.graph_features_dic['max_eigenvector_centrality'] = self.max_eigenvector_centrality()
        self.graph_features_dic['mean_eigenvector_centrality'] = self.mean_eigenvector_centrality()
        self.graph_features_dic['var_eigenvector_centrality'] = self.var_eigenvector_centrality()
        self.graph_features_dic['median_eigenvector_centrality'] = self.median_eigenvector_centrality()
        self.graph_features_dic['eigenvector_centrality_calc_time'] = self.eigenvector_centrality_calc_time
        self.graph_features_dic['min_pagerank_centrality'] = self.min_pagerank_centrality()
        self.graph_features_dic['max_pagerank_centrality'] =self.max_pagerank_centrality()
        self.graph_features_dic['mean_pagerank_centrality'] = self.mean_pagerank_centrality()
        self.graph_features_dic['var_pagerank_centrality'] = self.var_pagerank_centrality()
        self.graph_features_dic['median_pagerank_centrality'] = self.median_pagerank_centrality()
        self.graph_features_dic['pagerank_centrality_calc_time'] = self.pagerank_centrality_calc_time
        self.graph_features_dic['max_degrees_max_spanning_tree'] = self.calc_max_degrees_max_spanning_tree()
        self.graph_features_dic['min_degrees_max_spanning_tree'] = self.calc_min_degrees_max_spanning_tree()
        self.graph_features_dic['mean_degrees_max_spanning_tree'] = self.calc_mean_degrees_max_spanning_tree()
        self.graph_features_dic['var_degrees_max_spanning_tree'] = self.calc_var_degrees_max_spanning_tree()
        self.graph_features_dic['median_degrees_max_spanning_tree'] = self.calc_median_degrees_max_spanning_tree()
        self.graph_features_dic['max_spanning_tree_calc_time'] = self.max_spanning_tree_calc_time
        self.graph_features_dic['global_clust_coeff'] = self.global_clust_coeff
        self.graph_features_dic['global_clust_coeff_calc_time'] = self.global_clust_coeff_calc_time        
        self.graph_features_dic['min_farness_centrality'] = self.min_farness_centrality()
        self.graph_features_dic['max_farness_centrality'] = self.max_farness_centrality()
        self.graph_features_dic['var_farness_centrality'] = self.var_farness_centrality()
        self.graph_features_dic['mean_farness_centrality'] = self.mean_farness_centrality()
        self.graph_features_dic['median_farness_centrality'] = self.median_farness_centrality()
        self.graph_features_dic['farness_centrality_calc_time'] = self.farness_centrality_calc_time
        self.graph_features_dic['min_shortest_path_length_LCC'] = self.min_shortest_path_length_LCC()
        self.graph_features_dic['max_shortest_path_length_LCC'] = self.max_shortest_path_length_LCC()
        self.graph_features_dic['var_shortest_path_length_LCC'] = self.var_shortest_path_length_LCC()
        self.graph_features_dic['median_shortest_path_length_LCC'] = self.median_shortest_path_length_LCC()
        self.graph_features_dic['mean_shortest_path_length_LCC'] = self.mean_shortest_path_length_LCC()
        self.graph_features_dic['shortest_path_length_LCC_calc_time'] = self.shortest_path_lengths_LCC_calc_time
        self.graph_features_dic['degree_assortativity'] = self.degree_assortativity
        self.graph_features_dic['degree_assortativity_calc_time'] = self.degree_assortativity_calc_time


    def const_feature_vector(self, features_list):
        features_vector = torch.zeros(self.feature_num, 1)
        feature_indx = 0
        for feature in features_list:
            features_vector[feature_indx] = self.graph_features_dic[feature]
            feature_indx += 1

        return features_vector


    def calc_max_edge_nums(self, node_nums):
        return node_nums * (node_nums - 1) / 2


    def calc_max_var_degree(self, max_node_nums):
        return (max_node_nums - 1) ** 2


    def calc_max_min_features(self, max_node_nums):
        self.min_node_nums = 1
        self.max_node_nums = max_node_nums

        self.max_edge_nums = self.calc_max_edge_nums(max_node_nums)
        self.min_edge_nums = 0

        self.max_min_degree = self.max_max_degree = self.max_mean_degree = max_node_nums - 1
        self.min_min_degree = self.min_max_degree = self.min_mean_degree = 0

        self.max_var_degree = self.calc_max_var_degree(max_node_nums)
        self.min_var_degree = 0

        self.max_median_degree = max_node_nums - 1
        self.min_median_degree = 0

        self.max_graph_density = 1
        self.min_graph_density = 0

        self.max_min_clust_coeff = self.max_max_clust_coeff = self.max_mean_clust_coeff = self.max_median_clust_coeff = self.max_var_clust_coeff = 1
        self.min_min_clust_coeff = self.min_max_clust_coeff = self.min_mean_clust_coeff = self.min_median_clust_coeff = self.min_var_clust_coeff = 0

        self.max_min_node_betweenness_centrality = self.max_max_node_betweenness_centrality = self.max_mean_node_betweenness_centrality = self.max_median_node_betweenness_centrality = (max_node_nums - 1) * (max_node_nums - 2) / 2
        self.min_min_node_betweenness_centrality = self.min_max_node_betweenness_centrality = self.min_mean_node_betweenness_centrality = self.min_median_node_betweenness_centrality = 0
        self.max_var_node_betweenness_centrality = (max_node_nums - 1) * (max_node_nums - 2) / (2 * (max_node_nums))
        self.min_var_node_betweenness_centrality = 0

        self.max_min_edge_betweenness_centrality = self.max_max_edge_betweenness_centrality = self.max_mean_edge_betweenness_centrality = self.max_median_edge_betweenness_centrality = (max_node_nums - 1) * (max_node_nums - 2) / 2
        self.min_min_edge_betweenness_centrality = self.min_max_edge_betweenness_centrality = self.min_mean_edge_betweenness_centrality = self.min_median_edge_betweenness_centrality = 0
        self.max_var_edge_betweenness_centrality = (max_node_nums - 1) * (max_node_nums - 2) / (2 * (max_node_nums))
        self.min_var_edge_betweenness_centrality = 0

        self.max_num_connected_components = max_node_nums
        self.min_num_connected_components = 0

        self.max_node_nums_div_edge_nums = max_node_nums
        self.min_node_nums_div_edge_nums = 2 / (max_node_nums - 1)

        self.max_edge_nums_div_node_nums = (max_node_nums - 1) / 2
        self.min_edge_nums_div_node_nums = 0

        self.min_min_pagerank_centrality = self.min_max_pagerank_centrality = self.min_mean_pagerank_centrality = self.min_var_pagerank_centrality = self.min_median_pagerank_centrality =  0
        self.max_min_pagerank_centrality = self.max_max_pagerank_centrality = self.max_mean_pagerank_centrality = 10
        self.max_var_pagerank_centrality = 100
        self.max_median_pagerank_centrality = 10

        self.min_clust_coeff_calc_time = 0
        self.max_clust_coeff_calc_time = 3600 # assuming it will take at most 1 hour

        self.min_node_edge_betweenness_centrality_calc_time = 0
        self.max_node_edge_betweenness_centrality_calc_time = 3600 # assuming it will take at most 1 hour

        self.min_edge_betweenness_centrality_calc_time = 0
        self.max_edge_betweenness_centrality_calc_time = 3600  # assuming it will take at most 1 hour

        self.min_connected_components_calc_time = 0
        self.max_connected_components_calc_time = 3600 # assuming it will take at most 1 hour

        self.min_pagerank_centrality_calc_time = 0
        self.max_pagerank_centrality_calc_time = 3600 # assuming it will take at most 1 hour

        self.min_min_connected_component_size = 1
        self.max_min_connected_component_size = max_node_nums

        self.min_max_connected_component_size = 1
        self.max_max_connected_component_size = max_node_nums

        self.min_mean_connected_component_size = 1
        self.max_mean_connected_component_size = max_node_nums

        self.min_median_connected_component_size = 1
        self.max_median_connected_component_size = max_node_nums

        self.min_var_connected_component_size = 0
        self.max_var_connected_component_size = (max_node_nums - 2) ** 2 #not sure if it is correct

        self.min_modularity = 0
        self.max_modularity = 0.5 # Not sure

        self.min_min_community_size = self.min_max_community_size = self.min_mean_community_size = self.min_median_community_size = 1
        self.min_var_community_size = 0
        self.max_min_community_size = self.max_max_community_size = self.max_mean_community_size = self.max_median_community_size = max_node_nums
        self.max_var_community_size = (max_node_nums - 1) ** 2 # This is just an upper bound for the variance of communities sizes.

        self.min_max_degree_spanning_tree = 0
        self.max_max_degree_spanning_tree = max_node_nums - 1

        self.min_min_degree_spanning_tree = 0
        self.max_min_degree_spanning_tree = 1 # not sure...

        self.min_mean_degree_spanning_tree = 0
        self.max_mean_degree_spanning_tree = max_node_nums - 1 # This is only an upper bound, not maximum of mean degrees in the spanning tree

        self.min_var_degree_spanning_tree = 0
        self.max_var_degree_spanning_tree = (max_node_nums - 2) ** 2 # not sure if this is correct

        self.min_global_clust_coeff_calc_time = 0
        self.max_global_clust_coeff_calc_time = 3600 # assuming it will take at most 1 hour

        self.min_global_clust_coeff = 0
        self.max_global_clust_coeff = 1

        self.min_diameter = 0
        self.max_diameter = max_node_nums - 1

        self.min_max_spanning_tree_calc_time = 0
        self.max_max_spanning_tree_calc_time = 3600 # assuming it will take at most 1 hour

        self.min_number_maximal_cliques = 1
        self.max_number_maximal_cliques = max_node_nums

        self.min_min_size_maximal_cliques = 1
        self.max_min_size_maximal_cliques = max_node_nums

        self.min_max_size_maximal_cliques = 1
        self.max_max_size_maximal_cliques = max_node_nums

        self.min_mean_size_maximal_cliques = 1
        self.max_mean_size_maximal_cliques = max_node_nums

        self.min_median_size_maximal_cliques = 1
        self.max_median_size_maximal_cliques = max_node_nums

        self.min_var_size_maximal_cliques = 0
        self.max_var_size_maximal_cliques = (max_node_nums - 2) ** 2 #not sure if it is correct

        self.min_maximal_cliques_calc_time = 0
        self.max_maximal_cliques_calc_time = 3600 # assuming it will take at most 1 hour

        self.min_min_eigenvector_centrality = self.min_max_eigenvector_centrality = self.min_mean_eigenvector_centrality = self.min_var_eigenvector_centrality = 0
        self.max_min_eigenvector_centrality = self.max_max_eigenvector_centrality = self.max_mean_eigenvector_centrality = self.max_var_eigenvector_centrality = 1 # Not sure

        self.min_eigenvector_centrality_calc_time = 0
        self.max_eigenvector_centrality_calc_time = 3600 # assuming it will take at most 1 hour

        self.min_diameter_calc_time = 0
        self.max_diameter_calc_time = 3600 # assuming it will take at most 1 hour

        self.min_diameter_largest_component = 0
        self.max_diameter_largest_component = max_node_nums - 1

        self.min_min_eccentricity = self.min_max_eccentricity = self.min_mean_eccentricity = self.min_var_eccentricity = self.min_median_eccentricity = 0
        self.max_min_eccentricity = (max_node_nums - 1) / 2
        self.max_max_eccentricity = max_node_nums - 1
        self.max_mean_eccentricity = max_node_nums - 1 # This is just an upper bound
        self.max_median_eccentricity = max_node_nums - 1 # This is just an upper bound
        self.max_var_eccentricity = max_node_nums - 1 # This is just an upper bound

        self.min_num_communities = 1
        self.max_num_communities = max_node_nums


if __name__=='__main__':
    g = nx.erdos_renyi_graph(5000, 0.1)
    max_possible_node_nums = 5000
    features_list = ['node_nums', 'edge_nums', 'min_degree', 'max_degree', 'mean_degree', 'var_degree', 'median_degree',
                     'graph_density', 'min_clust_coeff', 'max_clust_coeff', 'mean_clust_coeff', 'var_clust_coeff', 'median_clust_coeff', 'min_node_betweenness_centrality', 'max_node_betweenness_centrality', 'mean_node_betweenness_centrality', 'var_node_betweenness_centrality', 'median_node_betweenness_centrality',
                     'min_edge_betweenness_centrality', 'max_edge_betweenness_centrality', 'mean_edge_betweenness_centrality', 'var_edge_betweenness_centrality', 'median_edge_betweenness_centrality',
                     'min_community_size' , 'max_community_size' , 'mean_community_size' , 'var_community_size' , 'median_community_size' , 'num_communities', 'modularity',
                     'max_degrees_max_spanning_tree', 'min_degrees_max_spanning_tree','mean_degrees_max_spanning_tree', 'var_degrees_max_spanning_tree', 'diameter', 'entropy_degrees',
                     'min_eccentricity_centrality', 'max_eccentricity_centrality', 'mean_eccentricity_centrality', 'var_eccentricity_centrality', 'median_eccentricity_centrality', 'entropy_clust_coeff']
    features_num = len(features_list)
    gp = graph_feature_extractor()
    gp.set_graph(g)
    gp.initial_processing(features_num)
    gp.calc_max_min_features(max_possible_node_nums)
    g_features = gp.calc_graph_features_normal()
    graph_vector = gp.const_feature_vector(features_list)
    print(graph_vector)
