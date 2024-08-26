import sys
import random
import networkx as nx
import networkit as nk
import matplotlib.pyplot as plt
from typing import Union
sys.path.append('../../')
from littleballoffur.sampler import Sampler


NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph


class RandomWalkSampler(Sampler):
    r"""An implementation of node sampling by random walks. A simple random walker
    which creates an induced subgraph by walking around. `"For details about the
    algorithm see this paper." <https://ieeexplore.ieee.org/document/5462078>`_

    Args:
        number_of_nodes (int): Number of nodes. Default is 100.
        seed (int): Random seed. Default is 42.
    """

    def __init__(self, number_of_nodes: int = 100, seed: int = 42):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self._set_seed()

    def _create_initial_node_set(self, graph, start_node):
        """
        Choosing an initial node.
        """
        if start_node is not None:
            if start_node >= 0 and start_node < self.backend.get_number_of_nodes(graph):
                self._current_node = start_node
                self._sampled_nodes = set([self._current_node])
            else:
                raise ValueError("Starting node index is out of range.")
        else:
            self._current_node = random.choice(
                range(self.backend.get_number_of_nodes(graph))
            )
            self._sampled_nodes = set([self._current_node])
        self._start_node = self._current_node  # check if it works fine

    def _do_a_step(self, graph):
        """
        Doing a single random walk step.
        """
        if len(self.backend.get_neighbors(graph, self._current_node)) == 0:
            self._current_node = self.s
        self._current_node = self.backend.get_random_neighbor(graph, self._current_node)
        self._sampled_nodes.add(self._current_node)

    def sample(
        self, graph: Union[NXGraph, NKGraph], start_node: int = None
    ) -> Union[NXGraph, NKGraph]:
        """
        Sampling nodes with a single random walk.

        Arg types:
            * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.
            * **start_node** *(int, optional)* - The start node.

        Return types:
            * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
        """
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        self._create_initial_node_set(graph, start_node)
        while len(self._sampled_nodes) < self.number_of_nodes:
            self._do_a_step(graph)
        new_graph = self.backend.get_subgraph(graph, self._sampled_nodes)
        return new_graph

if __name__=='__main__':
    g = nx.erdos_renyi_graph(500, 0.001)
    print('orig graph ')
    print(g.edges)
    #nx.draw(g)
    #plt.show()
    rw_sampler = RandomWalkSampler(number_of_nodes=50)
    g_sample = rw_sampler.sample(g)
    print('sampled graph')
    print(g_sample.edges)
    print(len(g_sample.nodes))
    #nx.draw(g_sample)
