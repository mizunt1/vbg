import sys,os
sys.path.append(os.getcwd())

import pickle
import numpy as np

from numpy.random import default_rng
from gflownet_sl.utils.graph import sample_erdos_renyi_linear_gaussian, sample_erdos_renyi_linear_gaussian_3_nodes
from gflownet_sl.utils.sampling import sample_from_discrete, sample_from_linear_gaussian
from gflownet_sl.utils.exhaustive import (get_full_posterior,
    get_edge_log_features, get_path_log_features, get_markov_blanket_log_features)
from gflownet_sl.utils.metrics import return_file_paths

num_variables = 5
num_edges = 5
num_seeds = 20
num_samples = 500

for i in range(num_seeds):
    file_paths = return_file_paths(i, "results2", "data_res2", base_dir="/network/scratch/m/mizu.nishikawa-toomey/test")
    rng = default_rng(i)
    rng_test = default_rng(i+3000)
    graph = sample_erdos_renyi_linear_gaussian(
        num_variables=num_variables,
        num_edges=num_edges,
        loc_edges=0.0,
        scale_edges=1.0,
        obs_noise=0.1,
        rng=rng,
    block_small_theta=True
    )
    data = sample_from_linear_gaussian(
        graph,
        num_samples=num_samples,
        rng=rng
    )
    data_test = sample_from_linear_gaussian(
        graph,
        num_samples=num_samples,
        rng=rng_test
    )
        
    with open(file_paths["graph"], 'wb') as f:
        pickle.dump(graph, f)
    data.to_csv(file_paths["data"])
    data_test.to_csv(file_paths["data_test"])
    full_posterior = get_full_posterior(
        data, score='lingauss', verbose=True, prior_mean=0., prior_scale=1., obs_scale=1)
    with open(file_paths["true_post"], 'wb') as f:
        pickle.dump(full_posterior, f)
