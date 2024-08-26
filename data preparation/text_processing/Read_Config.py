import configparser
import ast

def read_config_file_data(config_file, input_type):
    config = configparser.RawConfigParser()
    config.read(config_file)

    if input_type=='train':
        ########################### Read Train Config ###############
        synthetic_graphs_dic = dict(config.items('generate_train'))
        graph_types_dic = dict(config.items('train_graph_types'))
        input_num_per_type_size_dic = dict(config.items('train_input_num_per_ts'))
        graphs_sizes_dic = dict(config.items('train_graphs_sizes'))
        graphs_sizes_ranges_dic = dict(config.items('train_graphs_sizes_ranges'))
        syn_params_dic = dict(config.items('train_syn_params'))
        read_real_dic = dict(config.items('read_real_train'))

        synthetic_graphs = eval(synthetic_graphs_dic['synthetic_train_graphs'])
        real_graphs = eval(read_real_dic['real_train_graphs'])
        input_num_per_type_size = int(input_num_per_type_size_dic['train_input_num_per_type_size'])

    else:  # input_type is test
        ########################### Read Test Config ###############
        synthetic_graphs_dic = dict(config.items('generate_test'))
        graph_types_dic = dict(config.items('test_graph_types'))
        input_num_per_type_size_dic = dict(config.items('test_input_num_per_ts'))
        graphs_sizes_dic = dict(config.items('test_graphs_sizes'))
        graphs_sizes_ranges_dic = dict(config.items('test_graphs_sizes_ranges'))
        syn_params_dic = dict(config.items('test_syn_params'))
        read_real_dic = dict(config.items('read_real_test'))

        synthetic_graphs = eval(synthetic_graphs_dic['synthetic_test_graphs'])
        real_graphs = eval(read_real_dic['real_test_graphs'])
        input_num_per_type_size = int(input_num_per_type_size_dic['test_input_num_per_type_size'])

    generated_graph_types = {}
    for graph_type in graph_types_dic:
        generated_graph_types[graph_type] = eval(graph_types_dic[graph_type])

    graphs_sizes = {}
    for i in range(len(graphs_sizes_dic)):
        size_range = 'graphs_range_size' + str(i + 1)
        graphs_sizes[size_range] = eval(graphs_sizes_dic[size_range])

    graphs_sizes_ranges = {}
    for i in range(len(graphs_sizes_ranges_dic)):
        size_range = 'graphs_range_size' + str(i + 1)
        range_list_str = graphs_sizes_ranges_dic[size_range]
        range_list = ast.literal_eval(range_list_str)
        graphs_sizes_ranges[size_range] = range_list

    generated_graphs_params = {}
    for graph_type in syn_params_dic:
        params_list_str = syn_params_dic[graph_type]
        params_list = ast.literal_eval(params_list_str)
        generated_graphs_params[graph_type] = params_list

    return synthetic_graphs, real_graphs, generated_graph_types, input_num_per_type_size, \
        graphs_sizes, graphs_sizes_ranges, generated_graphs_params


def read_config_file_synthetic_data(config_file, input_type):
    config = configparser.RawConfigParser()
    config.read(config_file)

    if input_type=='train':
        ########################### Read Train Config ###############
        synthetic_graphs_dic = dict(config.items('generate_train'))
        graph_types_dic = dict(config.items('train_graph_types'))
        input_num_per_type_size_dic = dict(config.items('train_input_num_per_ts'))
        graphs_sizes_dic = dict(config.items('train_graphs_sizes'))
        graphs_sizes_ranges_dic = dict(config.items('train_graphs_sizes_ranges'))
        syn_params_dic = dict(config.items('train_syn_params'))
        synthetic_graphs = eval(synthetic_graphs_dic['synthetic_train_graphs'])
        input_num_per_type_size = int(input_num_per_type_size_dic['train_input_num_per_type_size'])

    else:  # input_type is test
        ########################### Read Test Config ###############
        synthetic_graphs_dic = dict(config.items('generate_test'))
        graph_types_dic = dict(config.items('test_graph_types'))
        input_num_per_type_size_dic = dict(config.items('test_input_num_per_ts'))
        graphs_sizes_dic = dict(config.items('test_graphs_sizes'))
        graphs_sizes_ranges_dic = dict(config.items('test_graphs_sizes_ranges'))
        syn_params_dic = dict(config.items('test_syn_params'))
        synthetic_graphs = eval(synthetic_graphs_dic['synthetic_test_graphs'])
        input_num_per_type_size = int(input_num_per_type_size_dic['test_input_num_per_type_size'])

    generated_graph_types = {}
    for graph_type in graph_types_dic:
        generated_graph_types[graph_type] = eval(graph_types_dic[graph_type])

    graphs_sizes = {}
    for i in range(len(graphs_sizes_dic)):
        size_range = 'graphs_range_size' + str(i + 1)
        graphs_sizes[size_range] = eval(graphs_sizes_dic[size_range])

    graphs_sizes_ranges = {}
    for i in range(len(graphs_sizes_ranges_dic)):
        size_range = 'graphs_range_size' + str(i + 1)
        range_list_str = graphs_sizes_ranges_dic[size_range]
        range_list = ast.literal_eval(range_list_str)
        graphs_sizes_ranges[size_range] = range_list

    generated_graphs_params = {}
    for graph_type in syn_params_dic:
        params_list_str = syn_params_dic[graph_type]
        params_list = ast.literal_eval(params_list_str)
        generated_graphs_params[graph_type] = params_list

    return synthetic_graphs, generated_graph_types, input_num_per_type_size, \
        graphs_sizes, graphs_sizes_ranges, generated_graphs_params


def read_config_file_synthetic_data_v2(config_file, input_type):
    '''
    This function reads synthtic data including graph features: graph_types, node_nums, densities and other info 
    '''
    config = configparser.RawConfigParser()
    config.read(config_file)

    if input_type=='train':
        ########################### Read Train Config ###############
        synthetic_graphs_dic = dict(config.items('generate_train'))
        graph_types_dic = dict(config.items('train_graph_types'))
        input_num_per_type_size_dic = dict(config.items('train_input_num_per_ts'))
        graphs_sizes_dic = dict(config.items('train_graphs_sizes'))
        graphs_sizes_ranges_dic = dict(config.items('train_graphs_sizes_ranges'))
        graphs_densities_dic = dict(config.items('train_syn_densities'))
        synthetic_graphs = eval(synthetic_graphs_dic['synthetic_train_graphs'])
        input_num_per_type_size = int(input_num_per_type_size_dic['train_input_num_per_type_size'])

    else:  # input_type is test
        ########################### Read Test Config ###############
        synthetic_graphs_dic = dict(config.items('generate_test'))
        graph_types_dic = dict(config.items('test_graph_types'))
        input_num_per_type_size_dic = dict(config.items('test_input_num_per_ts'))
        graphs_sizes_dic = dict(config.items('test_graphs_sizes'))
        graphs_sizes_ranges_dic = dict(config.items('test_graphs_sizes_ranges'))
        graphs_densities_dic = dict(config.items('test_syn_densities'))
        synthetic_graphs = eval(synthetic_graphs_dic['synthetic_test_graphs'])
        input_num_per_type_size = int(input_num_per_type_size_dic['test_input_num_per_type_size'])

    generated_graph_types = {}
    for graph_type in graph_types_dic:
        generated_graph_types[graph_type] = eval(graph_types_dic[graph_type])

    graphs_sizes = {}
    for i in range(len(graphs_sizes_dic)):
        size_range = 'graphs_range_size' + str(i + 1)
        graphs_sizes[size_range] = eval(graphs_sizes_dic[size_range])

    graphs_sizes_ranges = {}
    for i in range(len(graphs_sizes_ranges_dic)):
        size_range = 'graphs_range_size' + str(i + 1)
        range_list_str = graphs_sizes_ranges_dic[size_range]
        range_list = ast.literal_eval(range_list_str)
        graphs_sizes_ranges[size_range] = range_list

    generated_graphs_densities = {}
    graphs_densities_str = graphs_densities_dic['graph_densities']
    graphs_densities_list = ast.literal_eval(graphs_densities_str)

    return synthetic_graphs, generated_graph_types, input_num_per_type_size, \
        graphs_sizes, graphs_sizes_ranges, graphs_densities_list


def read_config_file_synthetic_data_v3(config_file, input_type):
    '''
    This function reads synthtic data including graph features: graph_types, node_nums, densities and other info 
    This function also includes additional graph types such as sbm, ff, 
    '''
    config = configparser.RawConfigParser()
    config.read(config_file)

    if input_type=='train':
        ########################### Read Train Config ###############
        synthetic_graphs_dic = dict(config.items('generate_train'))
        graph_types_dic = dict(config.items('train_graph_types'))
        input_num_per_type_size_dic = dict(config.items('train_input_num_per_ts'))
        graphs_sizes_dic = dict(config.items('train_graphs_sizes'))
        graphs_sizes_ranges_dic = dict(config.items('train_graphs_sizes_ranges'))
        graphs_densities_dic = dict(config.items('train_syn_densities'))
        synthetic_graphs = eval(synthetic_graphs_dic['synthetic_train_graphs'])
        input_num_per_type_size = int(input_num_per_type_size_dic['train_input_num_per_type_size'])
        stochastic_block_model_cluster_sizes_dic = dict(config.items('train_stochastic_block_model_cluster_sizes'))
        stochastic_block_model_probs_dic = dict(config.items('train_stochastic_block_model_probs'))
        powerlaw_cluster_dic = dict(config.items('train_powerlaw_cluster'))
        forest_fire_probs_dic = dict(config.items('train_forest_fire'))


    else:  # input_type is test
        ########################### Read Test Config ###############
        synthetic_graphs_dic = dict(config.items('generate_test'))
        graph_types_dic = dict(config.items('test_graph_types'))
        input_num_per_type_size_dic = dict(config.items('test_input_num_per_ts'))
        graphs_sizes_dic = dict(config.items('test_graphs_sizes'))
        graphs_sizes_ranges_dic = dict(config.items('test_graphs_sizes_ranges'))
        graphs_densities_dic = dict(config.items('test_syn_densities'))
        synthetic_graphs = eval(synthetic_graphs_dic['synthetic_test_graphs'])
        input_num_per_type_size = int(input_num_per_type_size_dic['test_input_num_per_type_size'])
        stochastic_block_model_cluster_sizes_dic = dict(config.items('test_stochastic_block_model_cluster_sizes'))
        stochastic_block_model_probs_dic = dict(config.items('test_stochastic_block_model_probs'))
        powerlaw_cluster_dic = dict(config.items('test_powerlaw_cluster'))
        forest_fire_probs_dic = dict(config.items('test_forest_fire'))


    generated_graph_types = {}
    for graph_type in graph_types_dic:
        generated_graph_types[graph_type] = eval(graph_types_dic[graph_type])

    graphs_sizes = {}
    for i in range(len(graphs_sizes_dic)):
        size_range = 'graphs_range_size' + str(i + 1)
        graphs_sizes[size_range] = eval(graphs_sizes_dic[size_range])

    graphs_sizes_ranges = {}
    for i in range(len(graphs_sizes_ranges_dic)):
        size_range = 'graphs_range_size' + str(i + 1)
        range_list_str = graphs_sizes_ranges_dic[size_range]
        range_list = ast.literal_eval(range_list_str)
        graphs_sizes_ranges[size_range] = range_list

    stochastic_block_model_cluster_sizes = {}
    for key in stochastic_block_model_cluster_sizes_dic:
        list_str = stochastic_block_model_cluster_sizes_dic[key]
        sizes_list = ast.literal_eval(list_str)
        stochastic_block_model_cluster_sizes[key] = sizes_list

    stochastic_block_model_probs = {}
    for key in stochastic_block_model_probs_dic:
        list_str = stochastic_block_model_probs_dic[key]
        sizes_list = ast.literal_eval(list_str)
        stochastic_block_model_probs[key] = sizes_list

    generated_graphs_densities = {}
    graphs_densities_str = graphs_densities_dic['graph_densities']
    graphs_densities_list = ast.literal_eval(graphs_densities_str)

    powerlaw_cluster_params = {}
    for key in powerlaw_cluster_dic:
        list_str = powerlaw_cluster_dic[key]
        params_list = ast.literal_eval(list_str)
        powerlaw_cluster_params[key] = params_list

    list_str = forest_fire_probs_dic['probs']
    forest_fire_probs = ast.literal_eval(list_str)

    return synthetic_graphs, generated_graph_types, input_num_per_type_size, \
        graphs_sizes, graphs_sizes_ranges, graphs_densities_list, stochastic_block_model_cluster_sizes, stochastic_block_model_probs, powerlaw_cluster_params, forest_fire_probs


def read_config_file_synthetic_data_v4(config_file, input_type):
    '''
    This function reads synthtic data including graph features: graph_types, node_nums, densities and other info 
    This function also includes additional graph types such as sbm, ff, 
    For SBM graphs this function only reads cluster numbers and densities. 
    '''
    config = configparser.RawConfigParser()
    config.read(config_file)

    if input_type=='train':
        ########################### Read Train Config ###############
        synthetic_graphs_dic = dict(config.items('generate_train'))
        graph_types_dic = dict(config.items('train_graph_types'))
        input_num_per_type_size_dic = dict(config.items('train_input_num_per_ts'))
        graphs_sizes_dic = dict(config.items('train_graphs_sizes'))
        graphs_sizes_ranges_dic = dict(config.items('train_graphs_sizes_ranges'))
        graphs_densities_dic = dict(config.items('train_syn_densities'))
        synthetic_graphs = eval(synthetic_graphs_dic['synthetic_train_graphs'])
        input_num_per_type_size = int(input_num_per_type_size_dic['train_input_num_per_type_size'])
        stochastic_block_model_cluster_nums_dic = dict(config.items('train_stochastic_block_model_cluster_nums'))
        stochastic_block_model_probs_ratios_dic = dict(config.items('train_stochastic_block_model_probs_ratios'))
        powerlaw_cluster_dic = dict(config.items('train_powerlaw_cluster'))
        forest_fire_probs_dic = dict(config.items('train_forest_fire_probs'))


    else:  # input_type is test
        ########################### Read Test Config ###############
        synthetic_graphs_dic = dict(config.items('generate_test'))
        graph_types_dic = dict(config.items('test_graph_types'))
        input_num_per_type_size_dic = dict(config.items('test_input_num_per_ts'))
        graphs_sizes_dic = dict(config.items('test_graphs_sizes'))
        graphs_sizes_ranges_dic = dict(config.items('test_graphs_sizes_ranges'))
        graphs_densities_dic = dict(config.items('test_syn_densities'))
        synthetic_graphs = eval(synthetic_graphs_dic['synthetic_test_graphs'])
        input_num_per_type_size = int(input_num_per_type_size_dic['test_input_num_per_type_size'])
        stochastic_block_model_cluster_nums_dic = dict(config.items('test_stochastic_block_model_cluster_nums'))
        stochastic_block_model_probs_ratios_dic = dict(config.items('test_stochastic_block_model_probs_ratios'))
        powerlaw_cluster_dic = dict(config.items('test_powerlaw_cluster'))
        forest_fire_probs_dic = dict(config.items('test_forest_fire_probs'))


    generated_graph_types = {}
    for graph_type in graph_types_dic:
        generated_graph_types[graph_type] = eval(graph_types_dic[graph_type])

    graphs_sizes = {}
    for i in range(len(graphs_sizes_dic)):
        size_range = 'graphs_range_size' + str(i + 1)
        graphs_sizes[size_range] = eval(graphs_sizes_dic[size_range])

    graphs_sizes_ranges = {}
    for i in range(len(graphs_sizes_ranges_dic)):
        size_range = 'graphs_range_size' + str(i + 1)
        range_list_str = graphs_sizes_ranges_dic[size_range]
        range_list = ast.literal_eval(range_list_str)
        graphs_sizes_ranges[size_range] = range_list

    # sbm params
    list_str = stochastic_block_model_cluster_nums_dic['cluster_nums']
    stochastic_block_model_cluster_nums = ast.literal_eval(list_str)
	
    list_str = stochastic_block_model_probs_ratios_dic['inter_intera_cluster_prob_ratios']
    stochastic_block_model_probs_ratios = ast.literal_eval(list_str)

    generated_graphs_densities = {}
    graphs_densities_str = graphs_densities_dic['graph_densities']
    graphs_densities_list = ast.literal_eval(graphs_densities_str)

    powerlaw_cluster_params = {}
    for key in powerlaw_cluster_dic:
        list_str = powerlaw_cluster_dic[key]
        params_list = ast.literal_eval(list_str)
        powerlaw_cluster_params[key] = params_list

    # forest fire params
    forest_fire_probs = {}
    for i in range(len(forest_fire_probs_dic)):
        size_range = 'graphs_range_size' + str(i + 1)
        list_str = forest_fire_probs_dic[size_range]
        forest_fire_probs[size_range] = ast.literal_eval(list_str)

    return synthetic_graphs, generated_graph_types, input_num_per_type_size, \
        graphs_sizes, graphs_sizes_ranges, graphs_densities_list, stochastic_block_model_cluster_nums, stochastic_block_model_probs_ratios, powerlaw_cluster_params, forest_fire_probs


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


def read_config_file_graph_features(config_file):
    config = configparser.RawConfigParser()
    config.read(config_file)
    graph_features_dic = dict(config.items('manual_graph_features'))
    graph_features_list = []

    for feature in graph_features_dic:
        if eval(graph_features_dic[feature]):
            graph_features_list.append(feature)

    return graph_features_list
