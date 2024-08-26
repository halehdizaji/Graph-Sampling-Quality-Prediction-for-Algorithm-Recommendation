import networkx as nx
import os
import pickle

def approximate_albert_barabasi_param(node_nums, graph_density):
    return round(node_nums * graph_density / 2)

def load_graphs(graphs_folder, input_graphs = {}):
    for entry in os.listdir(graphs_folder):
        if os.path.isfile(os.path.join(graphs_folder, entry)):
            graph_ID = entry[3:-7] #indexing separates ID_ from start of graph name and .pickle from the end of its name.
            print('graph id ',graph_ID)
            graph = pickle.load(
                open(graphs_folder + entry, 'rb'))
            input_graphs[graph_ID] = graph

    return input_graphs

def load_graph(graph_folder, graph_file_name, input_graphs = {}):
    graph_ID = graph_file_name[3:-7] #indexing separates ID_ from start of graph name and .pickle from the end of its name.
    print('graph id ',graph_ID)
    graph = pickle.load(
        open(graph_folder + graph_file_name, 'rb'))
    input_graph = (graph_ID, graph)

    return input_graph


if __name__=='__main__':
    node_nums = 1000
    density = 0.3
    approx_param = approximate_albert_barabasi_param(node_nums, density)
    g = nx.barabasi_albert_graph(node_nums, approx_param)
    real_den = 2 * len(g.edges())/(node_nums * (node_nums - 1))
    print('real density', real_den)
