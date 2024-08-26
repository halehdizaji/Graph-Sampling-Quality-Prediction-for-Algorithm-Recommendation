#!/bin/bash

dataset_num=1
################ Synthetic data folder #######
#root_folder=/home/itec/haleh/PycharmProjects/Graph_Sampling/data/syn_graphs

################## Real data folder 
root_folder=/home/itec/haleh/PycharmProjects/Graph_Sampling/data/real_graphs

dataset_folder=$root_folder/set_$dataset_num/graphs/

for f in "$dataset_folder"*;
do
	echo $(basename "${f}")
	features_folder=$root_folder/set_$dataset_num/features/$(basename "${f}")/
	mkdir -p $features_folder

	# CC
	python3 graph_processing/Extract_Clustering_Coeff_snap.py -graph_folder $dataset_folder -graph_file_name "$(basename "${f}")" -feature_folder $features_folder &
	# Connected comp
	#python3 graph_processing/Extract_Connected_Components_prop.py -graph_folder $dataset_folder -graph_file_name "$(basename "${f}")" -feature_folder $features_folder &
 	# Node & edge BC of snap
	python3 graph_processing/Extract_Betweenness_Centrality_snap.py -graph_folder $dataset_folder -graph_file_name "$(basename "${f}")" -feature_folder $features_folder &
	# approximate BC of Riodato
	python3 graph_processing/Extract_Approx_Node_Betweenness_Centrality_Riondatto.py -graph_folder $dataset_folder -graph_file_name "$(basename "${f}")" -feature_folder $features_folder &
	# EIC
	python3 graph_processing/Extract_eigenvector.py -graph_folder $dataset_folder -graph_file_name "$(basename "${f}")" -feature_folder $features_folder &
	# Global CC
	python3 graph_processing/Extract_Global_CC.py  -graph_folder $dataset_folder -graph_file_name "$(basename "${f}")" -feature_folder $features_folder & 
	# PR
	python3 graph_processing/Extract_PageRank.py -graph_folder $dataset_folder -graph_file_name "$(basename "${f}")" -feature_folder $features_folder &
	# Assor
	python3 graph_processing/Extract_assortativity.py -graph_folder $dataset_folder -graph_file_name "$(basename "${f}")" -feature_folder $features_folder &
	# MST
	python3 graph_processing/Extract_Max_Spanning_Tree.py -graph_folder $dataset_folder -graph_file_name "$(basename "${f}")" -feature_folder $features_folder &
	# shortest paths approx NK
	python3 graph_processing/Extract_approx_shortest_path_dist_stat_nk.py -graph_folder $dataset_folder -graph_file_name "$(basename "${f}")" -feature_folder $features_folder &
done


