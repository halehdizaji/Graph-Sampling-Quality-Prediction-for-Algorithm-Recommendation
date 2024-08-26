import sys
import random
import numpy as np
import networkx as nx
sys.path.append('../../')
from littleballoffur.sampler import Sampler


class FrontierSampler(Sampler):
    r"""An implementation of frontier sampling. A fixed number of random walkers
    traverses the graph and the walkers which make a step are selected randomly.
    The procedure might result in a disconnected graph as the walks might never
    connect with each other. `"For details about the algorithm see this paper." <https://www.cs.purdue.edu/homes/ribeirob/pdf/ribeiro_imc2010.pdf>`_

    Args:
        number_of_seeds (int): Number of seed nodes. Default is 10.
        number_of_nodes (int): Number of nodes to sample. Default is 100.
        seed (int): Random seed. Default is 42.
    """

    def __init__(
        self, number_of_seeds: int = 10, number_of_nodes: int = 100, seed: int = 42
    ):
        self.number_of_seeds = number_of_seeds
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self._set_seed()

    def _reweight(self, graph):
        """
        Create new seed weights.
        """
        self._seed_weights = [
            self.backend.get_degree(graph, seed) for seed in self._seeds
        ]
        weight_sum = np.sum(self._seed_weights)
        print('weight sum ', weight_sum)
        print('seed weights ', self._seed_weights)
        self._seed_weights = [
            float(weight) / weight_sum for weight in self._seed_weights
        ]
        print('seeds ', self._seeds)
        print('weights ', self._seed_weights)

    def _create_initial_seed_set(self, graph):
        """
        Choosing initial nodes.
        """
        nodes = self.backend.get_nodes(graph)
        self._seeds = random.sample(nodes, self.number_of_seeds)
        '''
        for seed in self._seeds:
            degree = self.backend.get_degree(graph, seed)
            if degree == 0:
            
        '''


    def _do_update(self, graph):
        """
        Choose new seed node.
        """
        sample = np.random.choice(self._seeds, 1, replace=False, p=self._seed_weights)[
            0
        ]
        index = self._seeds.index(sample)
        new_seed = random.choice(self.backend.get_neighbors(graph, sample))
        self._edges.add((sample, new_seed))
        self._nodes.add(sample)
        self._nodes.add(new_seed)
        self._seeds[index] = new_seed


    def sample(self, graph: nx.classes.graph.Graph, max_stucking_iter) -> nx.classes.graph.Graph:
        """
        Sampling nodes and edges with a frontier sampler.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be sampled from.

        Return types:
            * **new_graph** *(NetworkX graph)* - The graph of sampled nodes.
        """
        self._nodes = set()
        self._edges = set()
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        self._create_initial_seed_set(graph)
        no_improvement_iter = 0
        try:
            while len(self._nodes) < self.number_of_nodes:
                prev_sample_node_nums = len(self._nodes)
                print('len nodes ', len(self._nodes))
                self._reweight(graph)
                print('reweight done')
                self._do_update(graph)
                new_sample_node_nums = len(self._nodes)
                if prev_sample_node_nums == new_sample_node_nums:
                    no_improvement_iter += 1
                    if no_improvement_iter == max_stucking_iter:
                        break
                else:
                    no_improvement_iter = 0
            return self.backend.get_subgraph(self.backend.graph_from_edgelist(self._edges), self._nodes)
        except:
            return self.backend.get_subgraph(self.backend.graph_from_edgelist(self._edges), self._nodes)


if __name__=='__main__':
    node_nums = 1000
    max_stucking_iter = 50
    #graph1 = nx.barabasi_albert_graph(node_nums, 3)
    graph1 = nx.erdos_renyi_graph(node_nums, 0.001)
    sampling_percent = 0.1
    sample_node_nums = int(node_nums * sampling_percent)
    fs = FrontierSampler(number_of_seeds=10)
    fs.number_of_nodes = sample_node_nums
    sample_graph = fs.sample(graph1, max_stucking_iter)
    #sample_graph = fs.sample_corrected(graph1)
    print('sample graph nodes ' + str(sample_graph.nodes))
    print('num sampled nodes ', len(sample_graph.nodes))
    print('sample graph edges: ' + str(sample_graph.edges))
    # print(sample1.edges())
    print(nx.degree(graph1))
