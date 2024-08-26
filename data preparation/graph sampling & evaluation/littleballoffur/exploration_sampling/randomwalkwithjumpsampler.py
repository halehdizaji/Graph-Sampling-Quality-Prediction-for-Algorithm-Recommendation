import random
import networkx as nx
import networkit as nk
from typing import Union
from littleballoffur.sampler import Sampler

NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph


class RandomWalkWithJumpSampler(Sampler):
    r"""An implementation of node sampling by random walks with jumps.  The
    process is a discrete random walker on nodes which teleports back to a random
    node with a fixed probability. This might result in a  disconnected subsample
    from the original input graph. `"For details about the algorithm see this
    paper." <https://arxiv.org/abs/1002.1751>`_

    Args:
        number_of_nodes (int): Number of nodes. Default is 100.
        seed (int): Random seed. Default is 42.
        p (float): Jump (teleport) probability. Default is 0.1.
    """

    def __init__(self, number_of_nodes: int = 100, seed: int = 42, p: float = 0.1):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self.p = p
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

    def _do_a_step(self, graph):
        """
        Doing a single random walk step.
        """
        score = random.uniform(0, 1)
        if score < self.p:
            self._current_node = random.choice(
                range(self.backend.get_number_of_nodes(graph))
            )
        else:
            self._current_node = self.backend.get_random_neighbor(
                graph, self._current_node
            )
        self._sampled_nodes.add(self._current_node)

    def sample(
        self, graph: Union[NXGraph, NKGraph], start_node: int = None
    ) -> Union[NXGraph, NKGraph]:
        """
        Sampling nodes with a single random walk jumps.

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
    node_nums = 200
    #graph1 = nx.barabasi_albert_graph(node_nums, 3)
    graph1 = nx.erdos_renyi_graph(node_nums, 0.005)
    sampling_percent = 0.2
    sample_node_nums = int(node_nums * sampling_percent)
    rj = RandomWalkWithJumpSampler()
    sample_graph = rj.sample(graph1, start_node=3)
    print('sample graph nodes ' + str(sample_graph.nodes))
    print('num sampled nodes ', len(sample_graph.nodes))
    print('sample graph edges: ' + str(sample_graph.edges))
    # print(sample1.edges())
    print(nx.degree(graph1))