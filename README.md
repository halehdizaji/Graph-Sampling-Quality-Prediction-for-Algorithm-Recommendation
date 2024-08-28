# Graph Sampling Quality Prediction for Algorithm Recommendation


This repository includes the code and data for the **Graph Sampling Quality Prediction for Algorithm Recommendation** paper, a machine learning (ML) based method for predicting quality and performance of different graph sampling algorithms of three categories (node-based, edge-based and traversal).

This tutorial includes instructions for graph data preparation, processing, sampling, model training and evaluation, providing model reproducibility. It consists of following sections:
1. Model description
2. Data generation
3. Graph feature extraction
4. Graph sampling and evaluation
5. Mutual information analysis
6. Feature selection and preparation
7. Data analysis
8. Model training
8. Model testing
9. Result analysis
10. ML explainability

## Sampling algorithms
Our model predicts and recommends graph sampling algorithms of three categories, based on quality and performance metrics. It considers twelve sampling algorithms listed below:
* Node-based sampling
   * Random node
   * Random degree node
* Edge-based sampling
   * Random edge
   * Rendom node-edge
   * Induced random edge
* Traversal sampling
   * Random jump
   * Metropolis Hastings random walk
   * SnowBall
   * Forest Fire
   * Frontier
   * Rank degree
   * Expansion
##  Sampling quality metrics
We consider three quality metrics for evaluating sampling algorithms: degree distribution, clustering coefficient distribution and hop-plots distribution divergence.
## Sampling performance metric
We consider execution time of graph sampling algorithms as performance metric.

## ML models
Our tool consist of three ML models:
* Random forest (RF)
* k nearest neighbour (kNN)
* Multi layer perceptron (MLP)

## Data generation

### Synthetic graphs - train

**Albert-Barabasi**
We calculate parameters of these graphs given the desired densities.
Albert-Barabasi parameter calculation:
NewEdgesPerNode = round(node_nums * graph_density / 2)

- #Nodes: 10,000 ~ 100,000
- Density: 0.00001, 0.00003, 0.00005, 0.00007, 0.00009, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001
- #Graphs: 
- Library: Networkx
	
**Watts-Strogatz**
Watts-Strogatz parameter calculation:
NeighbourNum = round(graph_density * (node_nums - 1))

- #Nodes: 10,000 ~ 100,000
- Density: 0.00001, 0.00003, 0.00005, 0.00007, 0.00009, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001
- #Graphs: 
- Library: Networkx
  
**Erdos-Renyi**
- #Nodes: 10,000 ~ 100,000
- Density: 0.00001, 0.00003, 0.00005, 0.00007, 0.00009, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001
- #Graphs: 
- Library: Networkx

**PowerLawCluster**
We adjusted generation parameters (new edge numbers per node m, TriangleProb) to produce densities in  [0.00001, 0.001] and average clustering coefficient (CC) in [0.1, 0.6] extracted from real graph properties.

#Nodes | m | TriangleProb 
--- | --- | --- 
10,000 | 2 | 0.2, 0.3, 0.4, 0.5, 0.6
10,000 | 3 | 0.2, 0.4, 0.6, 0.8, 0.9, 1
10,000 | 5 | 0.4, 0.6, 0.8, 1
15,000 | 2 | 0.2, 0.5, 0.7
15,000 | 5 | 0.3, 0.7, 1
15,000 | 10 | 0.5, 1
20,000 | 2 | 0.2, 0.4, 0.6, 0.8
20,000 | 6 | 0.6, 0.8, 1
20,000 | 8 | 0.6, 1
20,000 | 10 | 0.5, 1
25,000 | 2 | 0.2, 0.4, 0.6, 0.8
25,000 | 5 | 0.4, 0.8, 1
25,000 | 15 | 0.7, 1
30,000 | 2 | 0.2, 0.4, 0.6, 0.8 
30,000 | 6 | 0.5, 1
30,000 | 15 | 0.8, 1
35,000 | 2 | 0.2, 0.4, 0.8
35,000 | 10 | 0.5, 1
35,000 | 20 | 0.9, 1
40,000 | 2 | 0.2, 0.5, 0.8
40,000 | 5 | 0.3, 0.5, 0.7, 1
40,000 | 10 | 0.7, 1
40,000 | 20 | 0.9, 1
45,000 | 2 | 0.2, 0.6, 0.8
45,000 | 10 | 0.9, 1
45,000 | 20 | 1

**Forest-Fire**
consists of 36 graphs with 10,000 ~ 100,000 nodes and adjusted forward/backward probabilites to produce densities in [0.00004, 0.001]. 
#Nodes | forward/backward probability 
--- | ---  
10,000 | 0.1, 0.2, 0.3
15,000 | 0.1, 0.2, 0.3
20,000 | 0.1, 0.2, 0.3
25,000 | 0.01, 0.1, 0.2, 0.3
30,000 | 0.01, 0.1, 0.2, 0.3
35,000 | 0.01, 0.1, 0.2, 0.3
40,000 | 0.01, 0.1, 0.2, 0.3
45,000 | 0.01, 0.1, 0.2, 0.3
50,000 | 0.01, 0.2, 0.3

**Stochastic Block Model**
 

### Synthetic graphs - test

### Real graphs

## Graph feature extraction


## Graph sampling and evaluation
## Mutual information analysis
## Feature selection and preparation
## Data analysis
## Model training
## Model testing
## Result analysis
## ML explainability
