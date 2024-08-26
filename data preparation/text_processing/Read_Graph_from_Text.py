import networkx as nx
import pandas as pd

def read_graph_from_text(filepath):
    graph = nx.Graph()
    with open(filepath) as fp:
        line = fp.readline()
        parts = line.split()
        node_nums = int(parts[1])
        graph.add_nodes_from(range(node_nums))
        line = fp.readline()
        parts = line.split()
        base = int(parts[1])
        line = fp.readline()
        while (len(line) > 0):
            parts = line.split()
            node_1 = int(parts[0]) - base
            node_2 = int(parts[1]) - base
            #graph.add_node(node_1)
            #graph.add_node(node_2)
            graph.add_edge(node_1, node_2)
            line = fp.readline()
    fp.close()
    return graph


def read_graph_from_text_no_self_loop(filepath):
    graph = nx.Graph()
    with open(filepath) as fp:
        line = fp.readline()
        parts = line.split()
        node_nums = int(parts[1])
        graph.add_nodes_from(range(node_nums))
        line = fp.readline()
        parts = line.split()
        base = int(parts[1])
        line = fp.readline()
        edges_set = set()
        while (len(line) > 0):
            parts = line.split()
            node_1 = int(parts[0]) - base
            node_2 = int(parts[1]) - base
            if (node_2, node_1) in edges_set:
                print('seen edge')
            else:
                edges_set.add((node_1, node_2))
            if node_1 == node_2:
                print('self loops at node ', node_2)
            else:
                graph.add_edge(node_1, node_2)
            #graph.add_node(node_1)
            #graph.add_node(node_2)
            line = fp.readline()
    fp.close()
    #print('seen edges ', edges_set)
    return graph


def read_graph_from_text_pd_no_self_loop(file_path):
    df = pd.read_csv(file_path)
    source_nodes_list = list(df['source_node'])
    target_nodes_list = list(df['target_node'])
    #print(source_nodes_list)
    #print(target_nodes_list)
    #print(int(df.iloc[0]['node_nums']))
    node_nums = int(df.iloc[0]['node_nums'])
    base = int(df.iloc[0]['base'])
    edges = []
    for i in range(len(source_nodes_list)):
        source_node = source_nodes_list[i]
        target_node = target_nodes_list[i]
        if not source_node == target_node:
            edges.append([source_node - base, target_node - base])

    graph = nx.Graph()
    graph.add_nodes_from(range(node_nums))
    graph.add_edges_from(edges)
    return graph


if __name__ == '__main__':
    #file_path = './Graph_Sampling_Alg_Selection/venv/data/Real_Data/txt files/test/bio-CE-GT.edges'
    #file_path = './Graph_Sampling_Alg_Selection/venv/data/Real_Data/txt files/test/econ-beacxc.mtx'
    #file_path = './Graph_Sampling_Alg_Selection/venv/data/Real_Data/txt files/test/econ-wm1.mtx'
    #file_path = './Graph_Sampling_Alg_Selection/venv/data/Real_Data/txt files/test/Email-EU-Core'
    file_path = './Graph_Sampling_Alg_Selection/venv/data/Real_Data/txt files/test/econ-beaflw.mtx'
    graph1 = read_graph_from_text(file_path)
    print(graph1.nodes)
    print(graph1.edges)
