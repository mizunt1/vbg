import sys, os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import pickle
import networkx as nx
import scipy.stats as st
import argparse
import sklearn as sk
from dibs.graph_utils import elwise_acyclic_constr_nograd
from gflownet_sl.metrics.metrics import expected_shd, expected_edges, threshold_metrics, get_mean_and_ci, LL
from gflownet_sl.utils.exhaustive import (get_full_posterior,
    get_edge_log_features, get_path_log_features, get_markov_blanket_log_features)
from gflownet_sl.utils.sampling import sample_from_linear_gaussian
from gflownet_sl.utils.graph import sample_erdos_renyi_linear_gaussian, sample_erdos_renyi_linear_gaussian_3_nodes
from gflownet_sl.utils.graph_plot import graph_to_matrix
from gflownet_sl.utils.metrics import posterior_estimate, get_log_features, return_file_paths
from gflownet_sl.utils.wandb_utils import return_ordered_data

def main(result_dir, seed, method, root_dir="/network/scratch/m/mizu.nishikawa-toomey/gflowdag/"):
    """
    calculates results for results for one seed and saves them in appropriate spaces in SCRATCH. 
    each result will be saved under
    /network/scratch/m/mizu.nishikawa-toomey/gflowdag/results{1,2}/<specified methodology>/seed/*.csv
    inputs:
    result_dir (str): either result1 or result2 depending on which experiments.
    seed (int) : which seed we are running our metrics on, each seed will have different data and results
    method (str): specify which method is used to infer posterior over g and theta, this code will create
    a directory if it doesnt exist and the results will be saved there
    est_post_g_name (str) : name of file of posterior samples. Should be saved as a npy file 
    note this posterior samples should be saved in -
    /network/scratch/m/mizu.nishikawa-toomey/gflowdag/results{1,2}/<specified methodology>/seed/name_of_file.npy  
    est_post_theta_name (str) : name of posterior samples. Should be saved as a npy file 
    note this posterior samples should be saved in - 
    /network/scratch/m/mizu.nishikawa-toomey/gflowdag/results{1,2}/<specified methodology>/seed/name_of_file.npy  
    """
    file_paths = return_file_paths(seed, result_dir, method, base_dir=root_dir)
    path_name_data = file_paths["data"]
    path_name_data_test = file_paths["data_test"]
    path_name_graph = file_paths["graph"]
    path_name_results = file_paths["results"]
    path_name_model = file_paths["model"]
    path_name_edge = file_paths["edge"]
    path_name_markov =  file_paths["markov"]
    path_name_path =  file_paths["path"]
    path_name_true_post = file_paths["true_post"]
    path_name_theta_params = file_paths["theta_params"]
    path_name_est_post_g = file_paths["est_post_g"]
    path_name_est_post_theta = file_paths["est_post_theta"]
    data = pd.read_csv(path_name_data, index_col=0)
    data_test = pd.read_csv(path_name_data_test, index_col=0)
    with open(path_name_graph, 'rb') as f:
        graph = pickle.load(f)
        # get data and graph from scratch and load them
    est_posterior_g = np.load(path_name_est_post_g)
    est_posterior_theta = np.load(path_name_est_post_theta)
    n_vars = est_posterior_g.shape[1]
    is_dag = elwise_acyclic_constr_nograd(est_posterior_g, n_vars) == 0
    if is_dag.sum() == 0:
        print("no dags")
        sys.exit()
    print("number_of_dags:", is_dag.sum())
    if method == "bcd":
        mask = est_posterior_theta > 0.3
        est_posterior_theta = (est_posterior_theta*[mask]).squeeze(0)

    est_posterior_g = est_posterior_g[is_dag, :, :]
    est_posterior_theta = est_posterior_theta[is_dag, :, :]
    
    # get posterior samples from posterior over graphs and parameters (if it exists)

    # Graph and theta samples should both be python arrays of size [d, n, n]
    # where d is number of samples, (1000) and n number of nodes (5 or 20) 

    ### results for results section 1 ###

    # negative marginal Log likelihood
    if method == "bcd":
        sigma_path = file_paths["sigma_bcd"]
        sigma_value = np.exp(np.load(sigma_path).mean())
        negll = -1*LL(est_posterior_g, est_posterior_theta, data_test.to_numpy(), sigma=sigma_value)
    else:
        negll = -1*LL(est_posterior_g, est_posterior_theta, data_test.to_numpy(), sigma=1)
    # using some dibs code to calculate likelihood metrics

    # SHD
    gt_adjacency = nx.to_numpy_array(graph, weight=None)
    # adjacency of true graph

    mean_shd = expected_shd(est_posterior_g, gt_adjacency)
    # calc mean_shd , ci95 not important
    # auroc
    thresholds = threshold_metrics(est_posterior_g, gt_adjacency)

    roc = thresholds['roc_auc']
    
    true_edge_weights = graph_to_matrix(graph, n_vars)
    weights_masked = est_posterior_theta * est_posterior_g
    rmse_theta = np.mean(abs(weights_masked - true_edge_weights))


    ### results for results section 2 ###
    if result_dir == "results2":
        # load full posterior here
        if method == "bcd":
            path_name_true_post = file_paths["true_post_bcd"]
        with open(path_name_true_post, 'rb') as tp:
            full_posterior = np.load(tp, allow_pickle=True)
        full_edge_log_features = get_edge_log_features(full_posterior)
        full_path_log_features = get_path_log_features(full_posterior)
        full_markov_log_features = get_markov_blanket_log_features(full_posterior)

        est_log_features = get_log_features(est_posterior_g, data.columns)
        # est_posterior is np.array of adjacency matrices of samples from the inferred (estimated) posterior.
        full_edge_ordered, est_edge_ordered = return_ordered_data(full_edge_log_features,
                                                                  est_log_features.edge, transform=np.exp)
        edge = {"full": full_edge_ordered, "est": est_edge_ordered}
        try:
            edge_mse = sk.metrics.mean_squared_error(est_edge_ordered, full_edge_ordered)
        except:
            return 
        df = pd.DataFrame(edge)
        df.to_csv(path_name_edge)
        edge_corr = np.corrcoef(full_edge_ordered, est_edge_ordered)[0][1]
        full_path_ordered, est_path_ordered = return_ordered_data(full_path_log_features,
                                                                  est_log_features.path, transform=np.exp)
        path_mse = sk.metrics.mean_squared_error(est_path_ordered, full_path_ordered)
        path = {"full": full_path_ordered, "est": est_path_ordered}
        df = pd.DataFrame(path)
        df.to_csv(path_name_path)
        path_corr = np.corrcoef(full_path_ordered, est_path_ordered)[0][1]

        full_markov_ordered, est_markov_ordered = return_ordered_data(full_markov_log_features,
                                                                      est_log_features.markov_blanket, transform=np.exp)
    
        markov = {"full": full_markov_ordered, "est": est_markov_ordered}

        markov_corr = np.corrcoef(full_markov_ordered, est_markov_ordered)[0][1]
        markov_mse = sk.metrics.mean_squared_error(est_markov_ordered, full_markov_ordered)
        df = pd.DataFrame(markov)
        df.to_csv(path_name_markov)

        # get edge, path and markov features of estimated posteriors as lists

        results = {"negll": negll, "mean_shd": mean_shd, "roc": roc,
                   "markov_corr": markov_corr, "edge_corr": edge_corr,
                   "path_corr": path_corr, "markov_mse": markov_mse,
                   "edge_mse": edge_mse, "path_mse": path_mse, "rmse_theta": rmse_theta}

    else:
        results = {"negll": negll, "mean_shd": mean_shd, "roc": roc, "rmse_theta": rmse_theta}
    df = pd.DataFrame(results, index=[0])

    df.to_csv(path_name_results)

def mean_ci_for_method(method_name, num_seeds, result_dir):
    """
    method_name (str): inference method name 
    num_seeds (int): number of seeds used in experiment
    num_nodes (int): number of nodes used in experiment
    """
    dfs = []
    summary_dict = {}
    summary_file = return_file_paths(1, result_dir, method_name)["summary"]
    for s in range(num_seeds):
        try:
            file_paths = return_file_paths(s, result_dir, method_name)
            result_df = pd.read_csv(file_paths["results"], index_col=0)
            dfs.append(result_df)
            print("done seed "+str(s))
        except:
            pass
    concat_df = pd.concat(dfs)
    for key in concat_df.keys():
        mean = concat_df[key].mean()
        ci = st.t.interval(
            alpha=0.95, df=len(concat_df[key])-1, loc=mean, scale=st.sem(concat_df[key].to_numpy()))
        max = concat_df[key].max()
        min = concat_df[key].min()
        summary_dict[key] = np.array([mean, ci[0], ci[1], max, min])
    df = pd.DataFrame.from_dict(summary_dict)
    df.to_csv(summary_file)

def collect_data_for_method(method_name, num_seeds, result_dir):
    """
    method_name (str): inference method name 
    num_seeds (int): number of seeds used in experiment
    num_nodes (int): number of nodes used in experiment
    """
    dfs = []
    summary_dict = {}
    summary_file = return_file_paths(1, result_dir, method_name)["summary_all_data"]
    for s in range(num_seeds):
        try:
            file_paths = return_file_paths(s, result_dir, method_name)
            result_df = pd.read_csv(file_paths["results"], index_col=0)
            dfs.append(result_df)
            print("done seed "+str(s))
        except:
            pass
    concat_df = pd.concat(dfs)
    for key in concat_df.keys():        
        summary_dict[key] = np.array(concat_df[key])
    df = pd.DataFrame.from_dict(summary_dict)
    df.to_csv(summary_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calc metrics')
    parser.add_argument('--method', type=str,
                        help='inference method for which metrics are being calculated, eg bcd')
    parser.add_argument('--num_seeds', type=int, default=20,
                        help='number of seeds which were run')
    parser.add_argument('--result_type', type=str,
                        choices=['results1', 'results2'], help='result type, results1 or results2')
    args = parser.parse_args()

    # calculates all benchmark quantities for all seeds within a given inference method
    for i in range(args.num_seeds):
        try:
            main(args.result_type, i, args.method)
        except:
            print("no result for seed " + str(i))
    mean_ci_for_method(args.method, args.num_seeds, args.result_type)
    collect_data_for_method(args.method, args.num_seeds, args.result_type)
    
