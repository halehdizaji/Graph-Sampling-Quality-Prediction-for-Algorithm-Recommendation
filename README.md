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

### Dataset details

#### Synthetic graphs - train
We generated five types of synthetic graphs, with the following settings.

**Albert-Barabasi**
These graphs consist of xxx graphs with 10,000 ~ 100,000 nodes, and densities in [0.00001, 0.001], generated with [Networkx](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html).
We calculate the new edge numbers per node parameter (NewEdgesPerNode) as follows (N: node numbers, D: graph density, m: new edge numbers per node):

$$m = \lfloor(N * D / 2)\rfloor$$
 
- m: 1~42
	
**Watts-Strogatz**
Watts-Strogatz graphs include xxx graphs with 10,000 ~ 100,000 nodes and densities in [0.00001, 0.001], generated with **Networkx**.
The following equation gives the number of neighbours of each node (|Nei|) in the ring toplogy, given N and D:

$$|Nei| = \lfloor(D * (N - 1))\rfloor$$

- $$|Nei|$$: 2~98
  
**Erdos-Renyi**
These graphs consist of xxx graphs with 10,000 ~ 100,000 nodes and edge probabilities (densities) in [0.00001, 0.001], generated with **Networkx**.

**PowerLawCluster**
PowerLawCluster graphs of **Networkx** include 60 graphs with adjusted parameters (new edge numbers per node m and TriangleProb) to produce densities in  [0.00001, 0.001] and average clustering coefficient (CC) in [0.1, 0.6] extracted from real graph properties. 

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
The evolution based Forest-Fire graphs consists of 36 graphs with 10,000 ~ 100,000 nodes and adjusted forward/backward probabilites to produce densities in [0.00004, 0.001]. We used [Snap-stanford](https://snap.stanford.edu/snappy/doc/reference/GenForestFire.html) library for their generation.

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
The clustering based graphs of stochastic block model consists of 180 graphs with the following characteristics, generated with **Networkx** ($K$: number of clusters, $\alpha$: inter-intra cluster probability ratio):

$$N$$: 5,000, 15,000, 25,000, 35,000, 45,000

$$D$$: 0.0001, 0.0005, 0.0009

$$K$$: 10, 15, 20, 25
$$\alpha$$: 0.01, 0.001, 0.1

We calculated parameters of these graphs (cluster size $$N_c$$, inter cluster density $$\rho'$$ and intra cluster density $$\rho$$) according to the following equations :

$$\rho = (D * (N-1) * K) / (N - K + \alpha * N / (K - 2))$$

$$\rho' = \alpha * \rho$$

$$N_c = \lfloor N/K\rfloor$$
 

#### Synthetic graphs - test
We generated 29 graphs of three types Albert-Barabasi, Erdos-Renyi and Watts-Strogatz with 150,000-450,000 nodes and following parameters:

* Albert-Barabasi with m of 1-22; 
* Watts-Strogatz with $$|Nei|$$ of 2-44;
* Erdos-Renyi with edge probabilities in [0.000008,0.0001].



#### Real graphs

### Data generation instructions
To generate synthetic graphs inside folder "data preparation/graph generation" run the following command:

```
bash run_generate_synthetic_graphs -dataset_num data_folder_number
```

It generates synthetic graphs with the characteristics definded in dataset_info file of the specified folder number.

## Graph feature extraction
This script extracts some time-consuming features i.e. node/edge betweenness, eigenvector, pagerank, clustering coefficient, components sizes, maximum spannig tree degrees, shortest path lengths and assortativity with their statistics and distributions.
### Execution instruction
The following command to extracts the major features of all graphs in the specified folder:

```
bash run_parallel_major_feature_extraction_multi_graphs
```

## Graph sampling and evaluation
It includes sampling from several graphs and evaluating samples under three quality metrics (degree, clustering coefficent and hop-plots distribution divergences) and execution time. 

### Execution instruction
The following command collects samples (in parallel) from the graphs in the specified folder with the desired sampling settings such as sampling algorithms and number of iterations:

```
bash sample_from_several_large_graphs
```
Then running this command evaluates samples under quality metrics.
```
bash run_parallel_processing_samples_multi_graph
```

## Mutual information analysis and feature selection
Mutual information (MI) analysis in notebook *samplings_analysis_MI.ipynb* uses sklearn for selecting the most relevant features for each metric (quality and performance), having MI scores higher than 0.99 with at least one sampling algorithm. 
Concatenation of these features with sampling features (12 dimensional 1-hot vector for sampling algorithm, 3 dimensions for 1-hot vector of algorithm type and 1 dimention for rate) constitutes the input feature vector for ML models.


## Feature normalization
We normalize all graphs features using maximum and exponential-logarithmic (EL) applied for the following statistics:
* **Maximum normalization**: minimum, average and medium values
* **EL normalization**: maximum and variance values, calculation times, and raw features.

The notebook *create_datasets.ipynb* includes the steps for normalization.

## Data analysis
The notebook *data_analysis.ipynb* includes data analysis steps including duplicate train/test data elimination and visualization of train/test data features using t-SNE.

## Model training
Model training consists of constructing feature vectors for each sampling metric and training three ML models, RF, MLP and kNN with tuning their hyper-parameters (see notebook *train_models.ipynb*).  

### Hyper-parameter tuning
We performed five-fold cross validation using GreadSearchCV with following search space for different hyper-parameters of ML models:

Model | Hyperparameter | Values
--- | ---  | ---  
kNN | algorithm | 'auto', 'ball_tree', 'kd_tree', 'brute'
kNN | n_neighbors | 4, 5, 10, 15
kNN | weights | 'uniform', 'distance'
RF | bootstrap | True
RF | max_depth | None, 90, 100, 110
RF | max_features | 2, 3
RF | min_samples_leaf | 2, 3, 4, 5
RF | min_samples_split | 8, 10, 12
RF | n_estimators | 100, 200, 300, 400
MLP | activation | 'tanh', 'relu'
MLP | hidden_layer_sizes | (30), (50), (100), (30,30), (50,50), (100,100)
MLP | solver | 'sgd', 'adam'
MLP | alpha | 0.0001, 0.05
MLP | learning_rate | 'constant', 'adaptive'
MLP | shuffle | True
MLP | early_stopping | True

## Model testing
## Result analysis
## ML explainability
