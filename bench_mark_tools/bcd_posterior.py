import sys,os
sys.path.append(os.getcwd())

import pickle
import numpy as np
import pandas as pd

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
    file_paths = return_file_paths(i, "results2", "bcd")
    data = pd.read_csv(file_paths["data"], index_col=0)
    scale = np.exp(np.load(file_paths["sigma_bcd"])).mean()    
    full_posterior = get_full_posterior(
        data, score='lingauss', verbose=True, prior_mean=0., prior_scale=1., obs_scale=scale)
    with open(file_paths["true_post_bcd"], 'wb') as f:
        pickle.dump(full_posterior, f)
