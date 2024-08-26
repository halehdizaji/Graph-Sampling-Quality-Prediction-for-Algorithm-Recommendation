#!/bin/sh

dataset_num=1
root_folder=./Graph_Sampling/data/real_graphs/set_$dataset_num
sampling_setting_num=5
dataset_folder=${root_folder}/graphs/temp/
#orig_graph_folder=$root_folder/graphs/

for f in "$dataset_folder"*;
do 
    python3 Process_Samplings_One_Graph.py -root_folder "${root_folder}" -dataset_folder "${dataset_folder}" -sampling_setting_num $sampling_setting_num -graph_file_name "$(basename "${f}")" &
done
