# Graph Sampling Quality Prediction for Algorithm Recommendation


This repository includes the code and data for the **Graph Sampling Quality Prediction for Algorithm Recommendation** paper, a machine learning (ML) based method for predicting quality and performance of different graph sampling algorithms of three categories (node-based, edge-based and traversal).

This tutorial includes instructions for graph data preparation, processing, sampling, model training and evaluation, providing model reproducibility. It consists of following sections:
1. Data generation
2. Graph feature extraction
3. Graph sampling and evaluation
4. Mutual information analysis
5. Feature selection and preparation
6. Data analysis
7. Model training
8. Model testing
9. Result analysis
10. ML explainability

## Data generation

### Synthetic graphs - train

Albert-Barabasi, Watts-Strogatz and ErdosRenyi:
	We calculate parameters of these graphs given the desired densities in Table 1.
Albert-Barabasi parameter calculation:
	

Watts-Strogatz parameter calculation:
watts_neighbour_num = round(graph_density * (node_nums - 1))

#Nodes: 10,000 ~ 100,000
Density: 0.00001, 0.00003, 0.00005, 0.00007, 0.00009, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001
#Graphs: 546
Library: Networkx

PowerLawCluster:
We adjusted generation parameters (new edge numbers per node m, TriangleProb) to produce densities in  [0.00001, 0.001] and average clustering coefficient (CC) in [0.1, 0.6] extracted from real graph properties.


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
