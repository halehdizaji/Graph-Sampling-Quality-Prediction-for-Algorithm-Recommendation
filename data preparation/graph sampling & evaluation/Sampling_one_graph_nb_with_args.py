#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pickle
import networkx as nx
import networkit as nk
import time
import os
import import_ipynb
import numpy as np
import matplotlib.pyplot as plt
import configparser
import argparse
import sys
import import_ipynb
import json as js
NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph
from typing import Union
from littleballoffur.sampler import Sampler
from littleballoffur.exploration_sampling.forestfiresampler import ForestFireSampler
#from littleballoffur.exploration_sampling.communitystructureexpansionsampler import CommunityStructureExpansionSampler
#import
#import sampling_algorithms as sa
#get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# # **Sampling Algorithms**

# MHRW

# In[13]:


import networkit as nk
from typing import Union
import sys
from littleballoffur.sampler import Sampler

NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph


class MetropolisHastingsRandomWalkSampler(Sampler):
    r"""An implementation of node sampling by Metropolis Hastings random walks.
    The random walker has a probabilistic acceptance condition for adding new nodes
    to the sampled node set. This constraint can be parametrized by the rejection
    constraint exponent. The sampled graph is always connected.  `"For details about the algorithm see this paper." <http://mlcb.is.tuebingen.mpg.de/Veroeffentlichungen/papers/HueBorKriGha08.pdf>`_

    Args:
        number_of_nodes (int): Number of nodes. Default is 100.
        seed (int): Random seed. Default is 42.
        alpha (float): Rejection constraint exponent. Default is 1.0.
    """

    def __init__(self, number_of_nodes: int = 100, seed: int = 42, alpha: float = 1.0):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self.alpha = alpha
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
        if len(list(nx.neighbors(graph, self._current_node))) == 0:
            #print('no neighbors')
            return
        new_node = self.backend.get_random_neighbor(graph, self._current_node)
        ratio = float(self.backend.get_degree(graph, self._current_node)) / float(
            self.backend.get_degree(graph, new_node)
        )
        ratio = ratio ** self.alpha
        if score < ratio:
            self._current_node = new_node
            self._sampled_nodes.add(self._current_node)

    def sample(
        self, graph: Union[NXGraph, NKGraph], max_stucking_iter = 50, start_node: int = None
    ) -> Union[NXGraph, NKGraph]:
        """
        Sampling nodes with a Metropolis Hastings single random walk.

        Arg types:
            * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.
            * **start_node** *(int, optional)* - The start node.

        Return types:
            * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled edges.
        """
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        self._create_initial_node_set(graph, start_node)
        stucking_iter = 0
        while len(self._sampled_nodes) < self.number_of_nodes:
            prev_sample_node_nums = len(self._sampled_nodes)
            #print('starting a step')
            self._do_a_step(graph)
            new_sample_node_nums = len(self._sampled_nodes)
            if prev_sample_node_nums == new_sample_node_nums:
                #print('no new sample')
                stucking_iter += 1
                if stucking_iter == max_stucking_iter:
                    break
            else:
                stucking_iter = 0

        new_graph = self.backend.get_subgraph(graph, self._sampled_nodes)
        return new_graph


# ## **RJ**

# In[14]:


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
        elif len(self.backend.get_neighbors(graph, self._current_node)) == 0:
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


# ## **SB**

# In[15]:


class Queue():
    # Constructor creates a list
    def __init__(self):
        self.queue = list()

    # Adding elements to queue
    def enqueue(self, data):
        # Checking to avoid duplicate entry (not mandatory)
        if data not in self.queue:
            self.queue.insert(0, data)
            return True
        return False

    # Removing the last element from the queue
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop()
        else:
            # plt.show()
            exit()

    # Getting the size of the queue
    def size(self):
        return len(self.queue)

    # printing the elements of the queue
    def printQueue(self):
        return self.queue


class Snowball():

    def __init__(self):
        self.G1 = nx.Graph()

    def snowball(self, G, size, k):
        q = Queue()
        list_nodes = list(G.nodes())
        m = k
        dictt = set()
        while(m):
            id = random.sample(list(G.nodes()), 1)[0]
            q.enqueue(id)
            m = m - 1
        # print(q.printQueue())
        while(len(self.G1.nodes()) <= size):
            #print('size graph: ', len(self.G1.nodes()))
            if(q.size() > 0):
                #print('here 1')
                print(q)
                id = q.dequeue()
                #print('id: ', id)
                self.G1.add_node(id)
                if(id not in dictt):
                    #print('here 2')
                    dictt.add(id)
                    list_neighbors = list(G.neighbors(id))
                    if(len(list_neighbors) > k):
                        #print('here 3')
                        for x in list_neighbors[:k]:
                            q.enqueue(x)
                            self.G1.add_edge(id, x)
                    elif(len(list_neighbors) <= k and len(list_neighbors) > 0):
                        #print('here 4')
                        for x in list_neighbors:
                            q.enqueue(x)
                            self.G1.add_edge(id, x)
                else:
                    continue
            else:
                print('restarting snowball')
                # modified line in snowball
                initial_nodes = random.sample([a for a in list(G.nodes()) if a not in list(dictt)], k)
                no_of_nodes = len(initial_nodes)
                for id in initial_nodes:
                    q.enqueue(id)
        return self.G1


# # **FS**
#Frontier Sampler
# In[16]:


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
        #print('weight sum ', weight_sum)
        #print('seed weights ', self._seed_weights)
        self._seed_weights = [
            float(weight) / weight_sum for weight in self._seed_weights
        ]
        #print('seeds ', self._seeds)
        #print('weights ', self._seed_weights)

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
                #print('len nodes ', len(self._nodes))
                self._reweight(graph)
                #print('reweight done')
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


# ## **Simple samplings**

# In[17]:


import random
import math
import networkx as nx
import numpy as np

def node_sampling_random(graph, sample_node_num):
    ### This function randomly samples nodes from a graph
    # and all edges between them.
    print('sampling random nodes')
    sampled_nodes = random.sample(graph.nodes(), sample_node_num)
    print('obtaining the subgraph')
    sampled_graph = graph.subgraph(sampled_nodes)

    return sampled_graph


def node_sampling_degree(graph, node_nums, sample_node_num):
    ### This function samples nodes from a graph according
      # to their degree distribution.
      # maybe the implementation is not efficient
    degrees = graph.degree()
    degrees_list = [degrees[i] for i in range(node_nums)]
    sampled_nodes = np.random.choice(node_nums, sample_node_num, p=[degrees_list[i]/sum(degrees_list) for i in range(node_nums)])
    sampled_graph = graph.subgraph(sampled_nodes)

    return sampled_graph

# Is the implementation of this sampler correct? Should we sample only edges?
def edge_sampling_random_v1(graph, sample_node_nums):
    ### This function samples random edges from a graph
    # until the required number of sample nodes are samples.
    graph_edges = list(graph.edges())
    sampled_nodes = set()
    num_sampled_nodes = 0
    sampled_edges = []
    while len(graph_edges) > 0 and num_sampled_nodes < sample_node_nums:
        sampled_edge_indx = random.randrange(len(graph_edges))
        sampled_edge = graph_edges.pop(sampled_edge_indx)
        sampled_edges.append(sampled_edge)
        sampled_nodes.add(sampled_edge[0])
        sampled_nodes.add(sampled_edge[1])
        num_sampled_nodes = len(sampled_nodes)

    sample_graph = nx.from_edgelist(sampled_edges)
    return sample_graph


def induced_edge_sampling(graph, sample_node_nums):
    ### This function samples random edges from a graph
    # until the required number of sample nodes are samples.
    graph_edges = list(graph.edges()).copy()
    sampled_nodes = set()
    num_sampled_nodes = 0
    sampled_edges = set()
    while len(graph_edges) > 0 and num_sampled_nodes < sample_node_nums:
        sampled_edge_indx = random.randrange(len(graph_edges))
        sampled_edge = graph_edges.pop(sampled_edge_indx)
        sampled_edges.add(sampled_edge)
        sampled_nodes.add(sampled_edge[0])
        sampled_nodes.add(sampled_edge[1])
        num_sampled_nodes = len(sampled_nodes)

    #print('sampled edges ', sampled_edges)
    #print('initial edge nums ', len(sampled_edges))
    sampled_nodes_list = list(sampled_nodes)
    num_sampled_nodes = len(sampled_nodes_list)
    for first_indx in range(num_sampled_nodes):
        previous = sampled_nodes_list[first_indx]
        for second_indx in range(first_indx+1,num_sampled_nodes):
            current = sampled_nodes_list[second_indx]
            #print(previous, current)
            if graph.has_edge(previous, current):
                sampled_edges.add((previous, current))

    sample_graph = nx.from_edgelist(list(sampled_edges))
    return sample_graph

def edge_sampling_random_v2(graph, node_sampling_percent, node_nums, ):
    ...

def random_node_edge_sampling(graph, sample_node_nums):
    '''
    :param graph:
    :param sample_node_nums:
    :return:
    This is the implementation of random node edge sampling.
    '''
    sampled_nodes = set()
    sampled_edges = []
    initial_graph_nodes = list(graph.nodes)
    while len(sampled_nodes) < sample_node_nums:
        sampled_node = random.sample(initial_graph_nodes, 1)[0]
        #print('sampled node ', sampled_node)
        sampled_nodes.add(sampled_node)
        #print('initial graph nodes ', initial_graph_nodes)
        initial_graph_nodes.remove(sampled_node)
        node_neighbors  = list(nx.neighbors(graph, sampled_node))
        if len(node_neighbors) > 0 and len(sampled_nodes) < sample_node_nums:
            sampled_neighbor = random.sample(node_neighbors, 1)[0]
            if not sampled_neighbor in sampled_nodes:
                sampled_nodes.add(sampled_neighbor)
                initial_graph_nodes.remove(sampled_neighbor)
            sampled_edges.append((sampled_node, sampled_neighbor))

    sample_graph = nx.from_edgelist(sampled_edges)
    sample_graph.add_nodes_from(sampled_nodes)

    return sample_graph


# In[24]:


class SRW_RWF_ISRW:

    def __init__(self):
        self.growth_size = 2
        self.T = 100    # number of iterations
        # with a probability (1-fly_back_prob) select a neighbor node
        # with a probability fly_back_prob go back to the initial vertex
        self.fly_back_prob = 0.15

    def random_walk_sampling_simple(self, complete_graph, nodes_to_sample):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample
        index_of_first_random_node = random.randint(0, nr_nodes - 1)
        sampled_graph = nx.Graph()

        sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'])

        iteration = 1
        edges_before_t_iter = 0
        curr_node = index_of_first_random_node
        while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            #print('edges: ' + str(edges))
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            #print('chosen node ' + str(chosen_node))
            sampled_graph.add_node(chosen_node)
            sampled_graph.add_edge(curr_node, chosen_node)
            curr_node = chosen_node
            iteration = iteration + 1

            if iteration % self.T == 0:
                if ((sampled_graph.number_of_edges() - edges_before_t_iter) < self.growth_size):
                    curr_node = random.randint(0, nr_nodes - 1)
                edges_before_t_iter = sampled_graph.number_of_edges()
        return sampled_graph

    def random_walk_sampling_with_fly_back(self, complete_graph, nodes_to_sample, fly_back_prob):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample

        index_of_first_random_node = random.randint(0, nr_nodes - 1)
        sampled_graph = nx.Graph()

        sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'])

        iteration = 1
        edges_before_t_iter = 0
        curr_node = index_of_first_random_node
        while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            sampled_graph.add_node(chosen_node)
            sampled_graph.add_edge(curr_node, chosen_node)
            choice = np.random.choice(['prev', 'neigh'], 1, p=[fly_back_prob, 1 - fly_back_prob])
            if choice == 'neigh':
                curr_node = chosen_node
            iteration = iteration + 1

            if iteration % self.T == 0:
                if ((sampled_graph.number_of_edges() - edges_before_t_iter) < self.growth_size):
                    curr_node = random.randint(0, nr_nodes - 1)
                    print("Choosing another random node to continue random walk ")
                edges_before_t_iter = sampled_graph.number_of_edges()

        return sampled_graph

    def random_walk_induced_graph_sampling(self, complete_graph, nodes_to_sample):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample
        index_of_first_random_node = random.randint(0, nr_nodes - 1)

        Sampled_nodes = set([complete_graph.nodes[index_of_first_random_node]['id']])

        iteration = 1
        nodes_before_t_iter = 0
        curr_node = index_of_first_random_node
        while len(Sampled_nodes) != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            Sampled_nodes.add(complete_graph.nodes[chosen_node]['id'])
            curr_node = chosen_node
            iteration = iteration + 1

            if iteration % self.T == 0:
                if ((len(Sampled_nodes) - nodes_before_t_iter) < self.growth_size):
                    curr_node = random.randint(0, nr_nodes - 1)
                nodes_before_t_iter = len(Sampled_nodes)

        sampled_graph = complete_graph.subgraph(Sampled_nodes)

        return sampled_graph


# # **RD**

# In[18]:


class RankDegree():

    def __init__(self):
        self.G1 = nx.Graph()

    def sample(self, G, size, seed_numbers, ro_neighbors, max_stucking_iter):
        G_local = G.copy()
        list_nodes = list(G.nodes())
        seeds = random.sample(list_nodes, seed_numbers)
        # print(q.printQueue())
        curr_sample_node_num = 0
        curr_sample_nodes = set()
        prev_sample_node_num = 0
        no_improvement = 0
        while(curr_sample_node_num <= size):
            print('current sample node num ', curr_sample_node_num)
            print('seeds ', seeds)
            selected_edges = []
            new_seeds = set()
            added_nodes = set()
            for id in seeds:
                list_neighbors = list(G_local.neighbors(id))
                print('len neighbors ', len(list_neighbors))
                node_degree_tuple = sorted(G_local.degree(list_neighbors), key=lambda x: x[1], reverse=True)
                sorted_list_neighbors = [nd[0] for nd in node_degree_tuple]
                k = int(ro_neighbors * len(list_neighbors))
                print('k ', k)
                if(len(list_neighbors) >= k):
                    print('len neighbors enough')
                    for x in sorted_list_neighbors[:k]:
                        new_seeds.add(x)
                        selected_edges.append((id, x))
                        added_nodes.add(id)
                        added_nodes.add(x)
                elif(len(list_neighbors) < k and len(list_neighbors) > 0):
                    print('len neighbors less')
                    x = sorted_list_neighbors[0]
                    new_seeds.add(x)
                    selected_edges.append((id, x))
                    added_nodes.add(id)
                    added_nodes.add(x)

            print('selected edges ', selected_edges)
            print('curr_sample_node_num ', curr_sample_node_num)
            new_nodes = [node for node in added_nodes if node not in curr_sample_nodes]
            if (len(new_nodes) + curr_sample_node_num) <= size:
                print('selected edges ', selected_edges)
                selected_edges = set(map(tuple, map(sorted, selected_edges)))
                print('selected edges ', selected_edges)
                for edge in selected_edges:
                    self.G1.add_edge(edge[0], edge[1])
                    G_local.remove_edge(edge[0], edge[1])
                    curr_sample_nodes.add(edge[0])
                    curr_sample_nodes.add(edge[1])
                seeds = new_seeds
                curr_sample_node_num = len(self.G1.nodes())
                seeds_degrees = list(dict(G_local.degree(seeds)).values())
                if sum(seeds_degrees) == 0 or len(seeds) == 0:
                    seeds = random.sample(list(G_local.nodes()), seed_numbers)
            else:
                sampled_new_nodes = random.sample(new_nodes, size - curr_sample_node_num)
                for edge in selected_edges:
                    if(edge[0] in sampled_new_nodes and edge[1] in sampled_new_nodes):
                        self.G1.add_edge(edge[0], edge[1])
                break
            if prev_sample_node_num == curr_sample_node_num:
                no_improvement += 1
                if no_improvement == max_stucking_iter:
                    break
            else:
                no_improvement = 0
            prev_sample_node_num = curr_sample_node_num

        return self.G1


# In[26]:


# expansion sampler

from typing import Union
from littleballoffur.sampler import Sampler

NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph


class CommunityStructureExpansionSampler(Sampler):
    r"""An implementation of community structure preserving expansion sampling.
    Starting with a random source node the procedure chooses a node which is connected
    to the already sampled nodes. This node is the one with the largest community expansion
    score. The extracted subgraph is always connected. `"For details about the algorithm see this paper." <http://arun.maiya.net/papers/maiya_etal-sampcomm.pdf>`_


    Args:
        number_of_nodes (int): Number of sampled nodes. Default is 100.
        seed (int): Random seed. Default is 42.
    """

    def __init__(self, number_of_nodes: int = 100, seed: int = 42):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self._set_seed()

    def _create_node_set(self, graph, start_node):
        """
        Choosing a seed node.
        """
        if start_node is not None:
            if start_node >= 0 and start_node < self.backend.get_number_of_nodes(graph):
                self._sampled_nodes = set([start_node])
            else:
                raise ValueError("Starting node index is out of range.")
        else:
            self._sampled_nodes = set(
                [random.choice(range(self.backend.get_number_of_nodes(graph)))]
            )

    def _make_target_set(self, graph):
        """
        Creating a new reshuffled frontier list of nodes.
        """
        self._targets = [
            neighbor
            for node in self._sampled_nodes
            for neighbor in self.backend.get_neighbors(graph, node)
        ]
        self._targets = list(set(self._targets).difference(self._sampled_nodes))
        random.shuffle(self._targets)

    def _choose_new_node(self, graph):
        """
        Choosing the node with the largest expansion.
        The randomization of the list breaks ties randomly.
        """
        largest_expansion = 0
        new_node = -1
        #print('target ', self._targets)
        for node in self._targets:
            #print('node is ',node)
            expansion = len(
                set(self.backend.get_neighbors(graph, node)).difference(
                    self._sampled_nodes
                )
            )
            if expansion >= largest_expansion:
                new_node = node
                largest_expansion = expansion
        if new_node== -1:
            self._sampled_nodes.add(random.choice(self._targets))          
        else:
            self._sampled_nodes.add(new_node)

    def sample(
        self, graph: Union[NXGraph, NKGraph], start_node: int = None
    ) -> Union[NXGraph, NKGraph]:
        """
        Sampling nodes iteratively with a community structure expansion sampler.

        Arg types:
            * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.
            * **start_node** *(int, optional)* - The start node.

        Return types:
            * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
        """
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        self._create_node_set(graph, start_node)
        while len(self._sampled_nodes) < self.number_of_nodes:
            self._make_target_set(graph)
            if len(self._targets) > 0:
                self._choose_new_node(graph)
            else:
                break
        new_graph = self.backend.get_subgraph(graph, self._sampled_nodes)
        return new_graph


# In[19]:


def run_sampling_algorithm(sampling_algorithm, graph, node_nums, sampling_percent, sample_node_nums, max_stucking_iter, object_RWI, object_Snowball, object_MHRW, object_RJ, object_FF, object_XS, object_FS, object_RD):
    if sampling_algorithm == 'random walk induced':
        start_time = time.time()
        sample = object_RWI.random_walk_induced_graph_sampling(graph,
                                                             sample_node_nums)  # graph, number of nodes to sample
        end_time = time.time()
        sample_node_nums = len(sample.nodes())
        print("Induced Subgraph Random Walk Sampling done.")
    elif sampling_algorithm == 'snowball':
        start_time = time.time()
        sample = object_Snowball.snowball(graph, sample_node_nums, 25)  # graph, number of nodes to sample , k set
        end_time = time.time()
        sample_node_nums = len(sample.nodes())
        print("Snowball Sampling done.")
    elif sampling_algorithm == 'random node':
        start_time = time.time()
        sample_node_nums = int(sampling_percent * node_nums)
        sample = node_sampling_random(graph, sample_node_nums)
        end_time = time.time()
        sample_node_nums = len(sample.nodes())
        print("Random node Sampling done.")
    elif sampling_algorithm == 'metropolis hastings random walk':
        start_time = time.time()
        sample_node_nums = int(sampling_percent * node_nums)
        object_MHRW.number_of_nodes = sample_node_nums
        sample = object_MHRW.sample(graph, max_stucking_iter = max_stucking_iter)
        end_time = time.time()
        sample_node_nums = len(sample.nodes())
        print("Metropolis Hastings Random Walk Sampling done.")
    elif sampling_algorithm == 'random degree node':
        start_time = time.time()
        sample_node_nums = int(sampling_percent * node_nums)
        sample = node_sampling_degree(graph, node_nums, sample_node_nums)
        end_time = time.time()
        sample_node_nums = len(sample.nodes())
        print("Random Degree Node Sampling done.")
    elif sampling_algorithm == 'random jump':
        start_time = time.time()
        object_RJ.number_of_nodes = sample_node_nums
        sample = object_RJ.sample(graph, sample_node_nums)
        end_time = time.time()
        sample_node_nums = len(sample.nodes())
        print("Random Jump Sampling done.")
    elif sampling_algorithm == 'random edge':
        start_time = time.time()
        sample_node_nums = int(sampling_percent * node_nums)
        sample = edge_sampling_random_v1(graph, sample_node_nums)
        end_time = time.time()
        sample_node_nums = len(sample.nodes())
        print("Random Edge Sampling done.")
    elif sampling_algorithm == 'random node edge':
        start_time = time.time()
        sample_node_nums = int(sampling_percent * node_nums)
        sample = random_node_edge_sampling(graph, sample_node_nums)
        end_time = time.time()
        sample_node_nums = len(sample.nodes())
        print("Random Node Edge Sampling done.")
    elif sampling_algorithm == 'forest fire':
        start_time = time.time()
        object_FF.number_of_nodes = sample_node_nums
        sample = object_FF.sample(graph)
        end_time = time.time()
        sample_node_nums = len(sample.nodes())
        print("Forest Fire Sampling done.")
    elif sampling_algorithm == 'induced random edge':
        start_time = time.time()
        sample = induced_edge_sampling(graph, sample_node_nums)
        end_time = time.time()
        sample_node_nums = len(sample.nodes())
        print("IRE done.")
    elif sampling_algorithm == 'expansion':
        start_time = time.time()
        object_XS.number_of_nodes = sample_node_nums
        sample = object_XS.sample(graph)
        end_time = time.time()
        sample_node_nums = len(sample.nodes())
        print("Expansion Sampling done.")
    elif sampling_algorithm == 'frontier':
        start_time = time.time()
        object_FS.number_of_nodes = sample_node_nums
        object_FS.number_of_seeds = 10
        sample = object_FS.sample(graph, max_stucking_iter)
        end_time = time.time()
        sample_node_nums = len(sample.nodes())
        print("Frontier Sampling done.")
    elif sampling_algorithm == 'rank degree':
        start_time = time.time()
        if node_nums < 500:
            s = 0.2
        elif node_nums < 10000:
            s = 0.1
        else:
            s = 0.01
        ro_rank_degree = 0.1
        seed_numbers = int(s * sample_node_nums)
        sample = object_RD.sample(graph, sample_node_nums, seed_numbers, ro_rank_degree, max_stucking_iter)
        end_time = time.time()
        sample_node_nums = len(sample.nodes())
        print("Rank Degree Sampling done.")



    running_time = end_time - start_time
    return sample, sample_node_nums, running_time



# In[20]:


def load_graph(graph_folder, graph_file_name):
    print('graph name:', graph_file_name)
    graph_ID = graph_file_name[3:-7] #indexing separates ID_ from start of graph name and .pickle from the end of its name.
    graph = pickle.load(
                open(graph_folder + graph_file_name, 'rb'))
    #print('graph nodes ', graph.nodes())
    input_graph = (graph_ID, graph)

    return input_graph


# used for large graphs
def perform_sampling_algorithms(input_graph, max_stucking_iter, object_RWI, object_Snowball, object_MHRW, object_RJ, object_FF, object_XS, object_FS, object_RD, sampling_experiment_folder):
    graphs_sampling_results = {}
    graph_ID = input_graph[0]
    graph = input_graph[1]
    print('graph ', graph_ID)
    node_nums = len(graph.nodes())
   ################################## Sampling ###############################
    for sampling_algorithm in sampling_algorithms:
        for sampling_percent in sampling_percents:
            print('sampling rate ', sampling_percent)
            Trial_ID = graph_ID + '_' + sampling_algorithm + '_' + str(sampling_percent)
            graphs_sampling_results[Trial_ID] = {}
            graphs_sampling_results[Trial_ID]['graph_ID'] = graph_ID
            graphs_sampling_results[Trial_ID]['sampling_percent'] = sampling_percent
            graphs_sampling_results[Trial_ID]['sampling_algorithm'] = sampling_algorithms_ids[sampling_algorithm]
            run_times = []
            for i in range(sampling_iter_num):
                sample_node_nums = int(sampling_percent * node_nums)
                print('starting sampling algorithm ', sampling_algorithm)
                sample_graph, sample_node_nums, run_time = run_sampling_algorithm(sampling_algorithm, graph, node_nums,
                                                                                  sampling_percent, sample_node_nums, max_stucking_iter, object_RWI,
                                                                                  object_Snowball, object_MHRW, object_RJ, object_FF, object_XS, object_FS, object_RD)
                pickle.dump(sample_graph, open(sampling_experiment_folder + graph_ID + '/sample_graphs/' + sampling_algorithm + '_' + str(sampling_percent) + '_sample_' + str(i), 'wb'))
                print("Number of nodes sampled=", len(sample_graph.nodes()))
                print("Number of edges sampled=", len(sample_graph.edges()))
                print('node num', sample_node_nums)  
                run_times.append(run_time)

            run_times = np.array(run_times)
            avg_run_time = run_times.mean()
            run_time_var = run_times.var()
            graphs_sampling_results[Trial_ID]['run_time'] = avg_run_time
            graphs_sampling_results[Trial_ID]['run_time var'] = run_time_var
            pickle.dump(graphs_sampling_results, open(sampling_experiment_folder + graph_ID + '/sampling_results', 'wb'))

    return graphs_sampling_results


# In[10]:


# read sampling config
def read_config_file_sampling_info(config_file):
    config = configparser.RawConfigParser()
    config.read(config_file)
    sampling_percents_dic = dict(config.items('sampling_percents'))
    sampling_algorithms_select_dic = dict(config.items('sampling_algorithms_select'))
    sampling_algorithms_ids_dic = dict(config.items('sampling_algorithms_ids'))
    sampling_iter_num_dic = dict(config.items('sampling_iter_num'))

    sampling_percents = {}
    for sampling_percent in sampling_percents_dic:
        sampling_percent_num = float(sampling_percent)
        sampling_percents[sampling_percent_num] = eval(sampling_percents_dic[sampling_percent])

    sampling_algorithms_select = {}
    for sampling_algorithm in sampling_algorithms_select_dic:
        sampling_algorithms_select[sampling_algorithm] = eval(sampling_algorithms_select_dic[sampling_algorithm])

    sampling_algorithms_ids = {}
    for sampling_algorithm in sampling_algorithms_ids_dic:
        sampling_algorithms_ids[sampling_algorithm] = int(sampling_algorithms_ids_dic[sampling_algorithm])

    sampling_iter_num = int(sampling_iter_num_dic['sampling_iter_num'])

    return sampling_percents, sampling_algorithms_select, sampling_algorithms_ids, sampling_iter_num


# In[37]:


# main of code
# inputs
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='My script')

    # Add arguments
    parser.add_argument('-sampling_setting_num', type=str, help='This is the number of sampling settings.', )
    parser.add_argument('-graph_file_name', type=str, help='This is the graph file name.')
    parser.add_argument('-dataset_num', type=str, help='This is the number of dataset.')
    parser.add_argument('-root_folder', type=str, help='This is the root folder.')
    parser.add_argument('-data_type', type=str, help='This is the data type.')
    parser.add_argument('-graph_folder', type=str, help='This is the graphs folder.')

    # Parse arguments
    args = parser.parse_args()

    sampling_setting_num = args.sampling_setting_num
    graph_file_name = args.graph_file_name 
    dataset_num = args.dataset_num
    root_folder = args.root_folder
    data_type = args.data_type
    graph_folder = args.graph_folder
    print('root_folder ', root_folder)
    sampling_experiment_folder = root_folder + '/data/' + data_type  + '_graphs/set_' + str(dataset_num) + '/samplings/setting_' + str(sampling_setting_num) + '/'
    sampling_config_file = root_folder + '/configs/samplings/setting_' + str(sampling_setting_num)
    config = configparser.RawConfigParser()
    config.read(sampling_config_file)

    ######################### Read from Config file #######################
    sampling_percents_dic, sampling_algorithms_select, sampling_algorithms_ids, sampling_iter_num = read_config_file_sampling_info(sampling_config_file)
    sampling_algorithms_ids_invrs = {v: k for k, v in sampling_algorithms_ids.items()}

    #graph_folder = root_folder + 'data/' + data_type  + '_graphs/set_' + str(dataset_num) + '/graphs/'

    sampling_algorithms = []

    print('start loading graph')
    input_graph = load_graph(graph_folder, graph_file_name)
    print('graph loaded')
    graph_ID = input_graph[0]
    graph = {input_graph[0]: input_graph[1]}
    object_RWI = SRW_RWF_ISRW()
    object_Snowball = Snowball()
    object_MHRW = MetropolisHastingsRandomWalkSampler()
    # object7 = TIES()
    object_RJ = RandomWalkWithJumpSampler()
    object_FF = ForestFireSampler()
    object_XS = CommunityStructureExpansionSampler()
    object_FS = FrontierSampler()
    object_RD = RankDegree()
    max_stucking_iter = 50

    for sampling_algorithm in sampling_algorithms_select.keys():
        if sampling_algorithms_select[sampling_algorithm]:
            sampling_algorithms.append(sampling_algorithm)

    # Check whether the specified path exists or not
    isExist = os.path.exists(sampling_experiment_folder + graph_ID + '/sample_graphs/')
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(sampling_experiment_folder + graph_ID + '/sample_graphs/')
        print("The new directory is created!")

    sampling_percents = []
    for sampling_percent in sampling_percents_dic:
        if sampling_percents_dic[sampling_percent]:
            sampling_percents.append(sampling_percent)

    graph_sampling_results = perform_sampling_algorithms(input_graph, max_stucking_iter, object_RWI, object_Snowball, object_MHRW, object_RJ, object_FF, object_XS, object_FS, object_RD, sampling_experiment_folder)
    pickle.dump(graph_sampling_results, open(sampling_experiment_folder + graph_ID + '/sampling_results', 'wb'))



