#!/bin/bash

sampling_setting_num=5
dataset_num=11
data_type='syn'
root_folder=/home/itec/haleh/PycharmProjects/Graph_Sampling

# segmented
dataset_folder=$root_folder/data/"$data_type"_graphs/set_$dataset_num/graphs/er/

#dataset_folder=$root_folder/data/"$data_type"_graphs/set_$dataset_num/graphs/


for f in "$dataset_folder"*;
do
        echo $(basename "${f}")
        python3 Sampling_one_graph_nb_with_args.py -sampling_setting_num $sampling_setting_num -graph_file_name "$(basename "${f}")" -dataset_num $dataset_num -root_folder $root_folder -data_type $data_type -graph_folder $dataset_folder &

done

