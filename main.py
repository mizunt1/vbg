import jax.numpy as jnp
import numpy as np
import pandas as pd
import networkx as nx
import optax
import wandb
import os
from time import time
import pickle

from pgmpy.utils import get_example_model
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from jax import random
from numpy.random import default_rng
import jax

from collections import namedtuple
from gflownet_sl.env import GFlowNetDAGEnv
from gflownet_sl.scores import BDeuScore, BGeScore
from gflownet_sl.gflownet import GFlowNet
from gflownet_sl.replay_buffer import ReplayBuffer
from gflownet_sl.utils.graph import sample_erdos_renyi_linear_gaussian, sample_erdos_renyi_linear_gaussian_3_nodes
from gflownet_sl.utils.sampling import sample_from_discrete, sample_from_linear_gaussian_interventions, sample_from_linear_gaussian, sample_from_linear_gaussian_int_het
from gflownet_sl.utils.jnp_utils import get_random_actions
from gflownet_sl.utils.io import save
from gflownet_sl.utils.wandb_utils import slurm_infos, table_from_dict, scatter_from_dicts, return_ordered_data
from gflownet_sl.utils.exhaustive import (get_full_posterior,
    get_edge_log_features, get_path_log_features, get_markov_blanket_log_features)
from gflownet_sl.utils.metrics import posterior_estimate, get_log_features, return_file_paths
from gflownet_sl.utils.interventional_sachs import download_sachs_intervention
from gflownet_sl.metrics.metrics import expected_shd, expected_edges, threshold_metrics, LL
from gflownet_sl.utils.gflownet import update_parameters_full, update_parameters
from gflownet_sl.utils.gflownet import compute_delta_score_lingauss_full, compute_delta_score_lingauss
from gflownet_sl.utils.gflownet import edge_marginal_means
from gflownet_sl.utils.graph_plot import graph_to_matrix


NormalParameters = namedtuple('NormalParameters', ['mean', 'precision'])
def main(args):
    wandb.init(
        project='interventions_test',
        settings=wandb.Settings(start_method='fork')
    )
    wandb.config.update(args)
    wandb.run.summary.update(slurm_infos())
    if args.num_variables <6:
        result = "results2"
    else:
        result = "results1"
    
    file_paths = return_file_paths(args.seed, result, "gflowdag", base_dir="/network/scratch/m/mizu.nishikawa-toomey/gflowdag")
    start_time = time()
    # Generate samples from a graph
    rng = default_rng(args.seed)
    rng_2 = default_rng(args.seed +1000)
    key = random.PRNGKey(args.seed)
    env_kwargs = dict()
    annot = True
    if args.hetero_noise: 
        obs_noise = random.uniform(key, minval=0.05, maxval=0.6, shape=(args.num_variables,))
    else:
        obs_noise = args.true_obs_noise
    if args.graph == 'erdos_renyi_lingauss':
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args.num_variables,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=args.scale_edges,
            low_edges=args.low_edges,
            obs_noise=obs_noise,
            rng=rng,
            block_small_theta=args.block_small_theta
        )
        data_test = sample_from_linear_gaussian(
            graph,
            num_samples=args.num_samples,
            rng=rng_2
        )
        if args.int_nodes != None:
            data_obs = sample_from_linear_gaussian(
                graph,
                np.ceil(args.num_samples*args.prop_obs).astype(int),
                rng=rng)

            data_int, int_mask = sample_from_linear_gaussian_int_het(
                graph,
                args.num_samples - np.ceil(args.num_samples*args.prop_obs).astype(int),
                args.int_nodes[0],
                rng=rng
            )
            frames = [data_obs, data_int]
            data = pd.concat(frames)
            int_mask_obs = np.full(data_obs.shape, False)
            int_mask = np.concatenate((int_mask_obs, int_mask))
            data_test.to_csv(os.path.join(wandb.run.dir, 'data_test.csv'))
        else:
            data_obs = sample_from_linear_gaussian(
                graph,
                args.num_samples,
                rng=rng)
            int_mask = np.full(data_obs.shape, False)
            data = data_obs
        wandb.save('data_test.csv', policy='now')
        plt.figure()
        plt.clf()
        annot = True
        if args.num_variables > 5:
            annot = False
        matrix = graph_to_matrix(graph, args.num_variables)
        true_graph = sns.heatmap(
            matrix, cmap="Blues", annot=annot, annot_kws={"size": 16})
        wandb.log({'true graph': wandb.Image(true_graph)})
        true_means = jnp.asarray(matrix)

    elif args.graph == 'erdos_renyi_lingauss_3_nodes':
        graph = sample_erdos_renyi_linear_gaussian_3_nodes(
            graph_index=args.graph_index,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
            rng=rng,
            block_small_theta=args.block_small_theta
        )
        data = sample_from_linear_gaussian(
            graph,
            num_samples=args.num_samples,
            rng=rng
        )
        plt.figure()
        plt.clf()
        matrix = graph_to_matrix(graph, args.num_variables)
        true_graph = sns.heatmap(
            matrix, cmap="Blues", annot=True, annot_kws={"size": 16})
        wandb.log({'true graph': wandb.Image(true_graph)})
        true_means = jnp.asarray(matrix)

    elif args.graph == 'sachs_discrete':
        graph = get_example_model('sachs')
        # Source: https://www.bnlearn.com/book-crc/
        data = pd.read_csv(
            'data/sachs.discretised.txt',
            delimiter=' ',
            dtype='category'
        )
        # env_kwargs = {'max_parents': 3}
    elif args.graph == 'sachs_continuous':
        graph = get_example_model('sachs')
        # Source: https://www.bnlearn.com/book-crc/
        data = pd.read_csv(
            'data/sachs.data.txt',
            delimiter='\t',
            dtype=np.float_
        )
        # Standardize data
        data = (data - data.mean()) / data.std()
    elif args.graph == 'sachs_intervention':
        graph = get_example_model('sachs')
        download_sachs_intervention()
        data = pd.read_csv(
            'data/sachs.interventional.txt',
            delimiter=' ',
            dtype='category'
        )
    elif args.graph == 'alarm':
        graph = get_example_model('alarm')
        data = sample_from_discrete(
            graph,
            num_samples=args.num_samples,
            rng=rng
        )
        env_kwargs = {'max_parents': 4}
    else:
        raise NotImplementedError(f'Unknown graph: {args.graph}')

    # Save the graph & data
    with open(os.path.join(wandb.run.dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(graph, f)
    wandb.save('graph.pkl', policy='now')

    data.to_csv(os.path.join(wandb.run.dir, 'data.csv'))
    wandb.save('data.csv', policy='now')

    # Create the environment
    if args.score == 'bge':
        scorer_cls = BGeScore
        scorer_kwargs = {
            'mean_obs': np.zeros(len(data.columns)),
            'alpha_mu': 1.,
            'alpha_w': len(data.columns) + 2.,
            **args.prior
        }
    elif args.score == 'bdeu':
        scorer_cls = BDeuScore
        scorer_kwargs = args.prior
    else:
        raise NotImplementedError('Score must be either "bdeu" or "bge".')
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        scorer=scorer_cls(data, **scorer_kwargs),
        num_workers=args.num_workers,
        context=args.mp_context,
        vb=args.vb,
        **env_kwargs
    )

    env_post = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        scorer=scorer_cls(data, **scorer_kwargs),
        num_workers=args.num_workers,
        context=args.mp_context,
        vb=args.vb,
        **env_kwargs
    )

    # Create the replay buffer
    replay = ReplayBuffer(
        args.replay_capacity,
        num_variables=env.num_variables,
        n_step=args.n_step,
        prioritized=args.replay_prioritized
    )

    # Create the GFlowNet & initialize parameters
    scheduler = optax.piecewise_constant_schedule(args.lr, {
        # int(0.4 * args.num_iterations): 0.1,
        # int(0.6 * args.num_iterations): 0.05,
        # int(0.8 * args.num_iterations): 0.01
    })

    gflownet = GFlowNet(
        optimizer=optax.adam(scheduler),
        delta=args.delta,
        n_step=args.n_step
    )
    params, state = gflownet.init(key, replay.dummy_adjacency)

    # Collect data (using random policy) to start filling the replay buffer
    observations = env.reset()
    indices = None
    for _ in trange(args.prefill, desc='Collect data'):
        # Sample random action
        actions, state = get_random_actions(state, observations['mask'])

        # Execute the actions and save the transitions to the replay buffer
        next_observations, rewards, dones, _ = env.step(np.asarray(actions))
        is_exploration = jnp.ones_like(actions)  # All these actions are from exploration step
        indices = replay.add(
            observations,
            actions,
            is_exploration,
            next_observations,
            rewards,
            dones,
            prev_indices=indices
        )
        observations = next_observations

    # Training loop
    tau = jnp.array(1.)  # Temperature for the posterior (should be equal to 1)
    epsilon = jnp.array(0.)
    num_samples = data.shape[0]
    data_introduced = 0
    first_run = True
    xtx = jnp.einsum('nk,nl->kl', data.to_numpy(), data.to_numpy())
    current_intervened_nodes = np.asarray([])
    if args.full_cov:
        prior = NormalParameters(
            mean=jnp.zeros((len(graph),)), precision=jnp.eye((len(graph))))
    else:
        prior = NormalParameters(mean=jnp.zeros((len(graph), len(graph))), precision=jnp.ones((len(graph), len(graph))))
    if args.vb:
        print("vb training")
        num_vb_updates = args.num_vb_updates
    else:
        num_vb_updates = 1
    with trange(args.num_iterations, desc='Training') as pbar:
        for iteration in pbar:
            losses = np.zeros(args.num_vb_updates)
            if (iteration + 1) % args.update_target_every == 0:
                # Update the parameters of the target network
                gflownet.set_target(params)
            if args.vb:
                # sample from posterior of graphs without adding to the environment
                if first_run:
                    for _ in range(100):
                        actions, is_exploration, next_state = gflownet.act(params, state, observations, epsilon)
                        next_observations, rewards, dones, _ = env.step(np.asarray(actions))
                        indices = replay.add(
                            observations,
                            actions,
                            is_exploration,
                            next_observations,
                            rewards,
                            dones,
                            prev_indices=indices
                        )
                        observations = next_observations
                        state = next_state
                        samples, subsq_mask = replay.sample(batch_size=args.batch_size, rng=rng)
                        params, state, logs = gflownet.step(
                            params,
                            gflownet.target_params,
                            state,
                            samples,
                            subsq_mask,
                            tau
                        )
                    orders = posterior_estimate(
                        params,
                        env_post,
                        state.key,
                        num_samples=args.num_samples_posterior
                    )
 
                    posterior_samples = (orders >= 0).astype(np.int_)
                    env_post.reset()
                    edge_params = prior
                    first_run = False
                else:
                    orders = posterior_estimate(
                        params,
                        env_post,
                        state.key,
                        num_samples=args.num_samples_posterior
                    )
                    posterior_samples = (orders >= 0).astype(np.int_)
       
                if args.full_cov:
                    new_edge_params = update_parameters_full(prior,
                                                             posterior_samples,
                                                             data_obs.to_numpy(),
                                                             obs_noise, edge_params)
                else:
                    edge_params = prior 
                    new_edge_params = update_parameters(edge_params, prior,
                                                        posterior_samples,
                                                        xtx,
                                                        obs_noise)

            else:
                edge_params = prior
                new_edge_params = edge_params

            if args.debug:
                if args.full_cov:
                    precision = jnp.stack([100*jnp.eye(args.num_variables) for i in range(args.num_variables)], axis=2)
                else:
                    precision = jnp.ones((args.num_variables, args.num_variables))
                new_edge_params = NormalParameters(mean=true_means, precision=precision)
            diff_mean = jnp.sum(abs(edge_params.mean - new_edge_params.mean)) / (edge_params.mean.shape[0]**2)
            diff_prec = jnp.sum(abs(edge_params.precision - new_edge_params.precision)) / (edge_params.mean.shape[0]**2)
            if args.vb:
                edge_params = new_edge_params
            for vb_iters in range(num_vb_updates):
                if args.vb:
                    if vb_iters == 0:
                        state = gflownet.reset_optim(state, params, key)
                        epsilon = jnp.array(0.)
                        # only update epsilon if we are half way through training
                    if iteration > (args.num_iterations * args.start_to_increase_eps):
                            if not args.keep_epsilon_constant:
                                epsilon = jnp.minimum(
                                    1-args.min_exploration,
                                    ((1-args.min_exploration)*2/num_vb_updates)*vb_iters)
                            else:
                                epsilon = jnp.array(0.)            
                else:
                    epsilon = jnp.minimum(1. - args.min_exploration, iteration*2 / args.num_iterations)

                try:
                    # Sample actions, execute them, and save transitions to the buffer
                    actions, is_exploration, next_state = gflownet.act(params, state, observations, epsilon)
                    next_observations, rewards, dones, _ = env.step(np.asarray(actions))
                    indices = replay.add(
                        observations,
                        actions,
                        is_exploration,
                        next_observations,
                        rewards,
                        dones,
                        prev_indices=indices
                    )
                    observations = next_observations
                    state = next_state
                except:
                    # Debug: If something went wrong in the environment, save the whole state
                    save(os.path.join(wandb.run.dir, 'debug.npz'),
                        params=params,
                        key=state.key,
                        epsilon=epsilon,
                        actions=actions,
                        **observations
                    )
                    wandb.save('debug.npz', policy='now')
                    raise

                samples, subsq_mask = replay.sample(batch_size=args.batch_size, rng=rng)
                if args.vb:
                    if args.full_cov:
                        diff_marg_ll = jax.vmap(
                            compute_delta_score_lingauss_full, in_axes=(0,0,None, None,
                                                                        None,None,
                                                                        None, None,None))(
                                samples['adjacency'][0],
                                samples['actions'][0],
                                edge_params,
                                prior,
                                data,
                                int_mask,
                                obs_noise,
                                args.weight,
                                args.use_erdos_prior)

                    else:
                        diff_marg_ll = jax.vmap(
                            compute_delta_score_lingauss, in_axes=(0,0,None,None,None,None))(
                                samples['adjacency'][0],
                                samples['actions'][0],
                                edge_params,
                                prior,
                                xtx,
                                obs_noise)
                    samples['rewards'][0] = diff_marg_ll
                    mean_rewards = jnp.mean(diff_marg_ll)
                params, state, logs = gflownet.step(
                    params,
                    gflownet.target_params,
                    state,
                    samples,
                    subsq_mask,
                    tau
                )
                time_elapsed = time() - start_time
                time_in_hrs = time_elapsed / (60*60)
                if args.vb:
                    losses[vb_iters] = logs['loss']
                    if args.plot_vb_iter: 
                        if (vb_iters+1) % (args.log_every) == 0:
                            gt_adjacency = nx.to_numpy_array(graph, weight=None)
                            edge_cov = jax.vmap(jnp.linalg.inv, in_axes=-1, out_axes=-1)(edge_params.precision)
                            mean_shd = expected_shd(posterior_samples, gt_adjacency)
                            posterior_theta = random.multivariate_normal(key,
                                                                             edge_params.mean,
                                                                             edge_cov, shape=(args.num_samples_posterior,args.num_variables))
                            log_like = -1*LL(posterior_samples, posterior_theta, data_test.to_numpy(), obs_noise)

                            wandb.log(
                                       {'vb iter loss': logs['loss'],
                                        'vb iter step': (iteration+1)*vb_iters,
                                        'delta mean': diff_mean,
                                        'delta_prec': diff_prec,
                                        'eps': epsilon,
                                        'mean delta': mean_rewards,
                                        'mean shd': mean_shd,
                                        'log_like': log_like,
                                       })

                replay.update_priorities(samples, logs['error'])
            if (iteration + 1) % (args.log_every * 10) == 0:
                errors = logs['error'][logs['mask'] == 1.]
                wandb.log({
                    'replay/rewards': wandb.Histogram(replay.transitions['rewards']),
                    'replay/scores': wandb.Histogram(replay.transitions['scores']),
                    'replay/num_edges': wandb.Histogram(replay.transitions['num_edges']),
                    'replay/is_exploration': np.mean(replay.transitions['is_exploration']),
                    'error/distribution-log10': wandb.Histogram(jnp.log10(jnp.abs(errors[errors > 0])))
                }, commit=False)
            if (iteration + 1) % args.log_every == 0:
                errors = logs['error'][logs['mask'] == 1.]
                if args.vb:
                    losses_min = losses.min()
                    losses_max = losses.max()
                    losses_last = losses[-1]
                    losses_first = losses[0]
                    loss_diff = losses_first - losses_last

                    plt.clf()
                    means_plot = sns.heatmap(
                        edge_params.mean, cmap="Blues", annot=annot, annot_kws={"size": 16})
                    wandb.log({'means': wandb.Image(means_plot)})
                    if args.plot_cov:
                        for i in range(args.num_variables):
                            plt.clf()
                            cov_plot = sns.heatmap(
                                jnp.linalg.inv(edge_params.precision[:,:,i]),
                                cmap="Blues", annot=True, annot_kws={"size": 16})
                            string = 'covariance '+ str(i)+' plot'
                            wandb.log(
                                {string: wandb.Image(cov_plot)})
                    wandb.log(
                        {'gf_loss_min': losses_min,
                         'gf_loss_max': losses_max,
                         'gf_loss_last': losses_last,
                         'gf_loss_diff': loss_diff,
                         'time in hours': time_in_hrs})
                    mat = [[(matrix[i,j] - edge_params.mean[i,j])**2 for j in range(args.num_variables)] for i in range(args.num_variables)]
                    sum_ = sum([sum(mat[i]) for i in range(args.num_variables)])
                    mean_squared_error_mean = sum_ / (args.num_variables**2)
                    wandb.log({'mse of mean': mean_squared_error_mean})
                    edge_marginal_means_ = edge_marginal_means(edge_params.mean, posterior_samples)
                    plt.clf()
                    edge_mm_plot = sns.heatmap(
                        edge_marginal_means_, cmap="Blues", annot=annot, annot_kws={"size": 16})
                    wandb.log({'edge marginal means': wandb.Image(edge_mm_plot)})
                    mat = [[(matrix[i,j] - edge_marginal_means_[i,j])**2 for j in range(args.num_variables)] for i in range(args.num_variables)]
                    sum_ = sum([sum(mat[i]) for i in range(args.num_variables)])
                    mean_squared_error_emm = sum_ / (args.num_variables**2)
                    wandb.log({'mse of edge marginal mean': mean_squared_error_emm})
                    edge_marginals = np.sum(posterior_samples, axis=0)/posterior_samples.shape[0]
                    plt.clf()
                    edge_m_plot = sns.heatmap(
                        edge_marginals, cmap="Blues", annot=annot, annot_kws={"size": 16})
                    wandb.log({'edge marginals': wandb.Image(edge_m_plot)})



                wandb.log({
                    'step': iteration,
                    'loss': logs['loss'],
                    'replay/size': len(replay),
                    'tau': tau,
                    'target/mse': logs['target/mse'],

                    'error/mean': jnp.abs(errors).mean(),
                    'error/max': jnp.abs(errors).max(),
                })
            pbar.set_postfix(loss=f"{logs['loss']:.2f}", epsilon=f"{epsilon:.2f}")

    # Save final model
    if args.benchmarking:
        save(file_paths["model"], params=params)
    wandb.save('model.npz', policy='now')

    # Evaluate the posterior estimate
    orders = posterior_estimate(
        params,
        env,
        state.key,
        num_samples=args.num_samples_posterior
    )
    posterior = (orders >= 0).astype(np.int_)
    if args.full_cov:

        edge_cov = jax.vmap(jnp.linalg.inv, in_axes=-1, out_axes=-1)(edge_params.precision)
    else: 
        edge_cov = jnp.linalg.inv(edge_params.precision)
    if args.full_cov:
        posterior_theta = jax.vmap(
            random.multivariate_normal, in_axes=(None, -1, -1, None),  out_axes=(-1))(key,
                                                                                edge_params.mean,
                                                                                edge_cov, (args.num_samples_posterior,))
    else:
        posterior_theta = rng.random.multivariate_normal(key,
                                                     edge_params.mean,
                                                     edge_cov, shape=(args.num_samples_posterior,args.num_variables))
        
    log_like = -1*LL(posterior, posterior_theta, data_test.to_numpy(), obs_noise)
    
    wandb.run.summary.update({"negative log like": log_like})
    if args.benchmarking:
        with open(file_paths["est_post_g"], 'wb') as f:
            np.save(f, posterior)
        with open(file_paths["est_post_theta"], 'wb') as f:
                np.save(f, posterior_theta)
        with open(file_paths["theta_params"], 'wb') as f:
            pickle.dump(edge_params, f)
            
    edge_marginals = np.sum(posterior, axis=0)/posterior.shape[0]
    plt.clf()
    edge_m_plot = sns.heatmap(
        edge_marginals, cmap="Blues", annot=annot, annot_kws={"size": 16})
    wandb.log({'edge marginals': wandb.Image(edge_m_plot)})

    # The posterior estimate returns the order in which the edges have been added
    log_features = get_log_features(posterior, data.columns)
    wandb.run.summary.update({
        'posterior/estimate/edge': table_from_dict(log_features.edge),
        'posterior/estimate/path': table_from_dict(log_features.path),
        'posterior/estimate/markov_blanket': table_from_dict(log_features.markov_blanket)
    })

    # Compute metrics on the posterior estimate
    gt_adjacency = nx.to_numpy_array(graph, weight=None)

    # Expected SHD
    mean_shd = expected_shd(posterior, gt_adjacency)

    # Expected # Edges
    mean_edges = expected_edges(posterior)

    # Threshold metrics
    thresholds = threshold_metrics(posterior, gt_adjacency)

    wandb.run.summary.update({
        'metrics/shd/mean': mean_shd,
        'metrics/edges/mean': mean_edges,
        'metrics/thresholds': thresholds
    })

    # Save posterior estimate
    with open(os.path.join(wandb.run.dir, 'posterior_estimate.npy'), 'wb') as f:
        np.save(f, posterior)
    wandb.save('posterior_estimate.npy', policy='now')
    with open(os.path.join(wandb.run.dir, 'posterior_estimate_thetas.npy'), 'wb') as f:
        np.save(f, posterior_theta)
    wandb.save('posterior_estimate_theta.npy', policy='now')

    
    # Evaluate: for small enough graphs, evaluate the full posterior
    if (args.graph in ['erdos_renyi_lingauss', 'erdos_renyi_lingauss_3_nodes']) and (args.num_variables < 6):
        # Default values set by data generation
        # See `sample_erdos_renyi_linear_gaussian` above
        if args.vb:
            full_posterior = get_full_posterior(
                data, score='lingauss', verbose=True, prior_mean=0., prior_scale=1., obs_scale=obs_noise)
        else:
            full_posterior = get_full_posterior(
                data, score='bge', verbose=True, **scorer_kwargs)
        # Save full posterior

        with open(os.path.join(wandb.run.dir, 'posterior_full.npz'), 'wb') as f:
            np.savez(f, log_probas=full_posterior.log_probas,
                **full_posterior.graphs.to_dict(prefix='graphs'),
                **full_posterior.closures.to_dict(prefix='closures'),
                **full_posterior.markov.to_dict(prefix='markov')
            )
        wandb.save('posterior_full.npz', policy='now')

        full_edge_log_features = get_edge_log_features(full_posterior)
        full_path_log_features = get_path_log_features(full_posterior)
        full_markov_log_features = get_markov_blanket_log_features(full_posterior)
        wandb.run.summary.update({
            'posterior/fufll/edge': table_from_dict(full_edge_log_features),
            'posterior/full/path': table_from_dict(full_path_log_features),
            'posterior/full/markov_blanket': table_from_dict(full_markov_log_features)
        })
        wandb.log({
            'posterior/scatter/edge': scatter_from_dicts('full', full_edge_log_features,
                'estimate', log_features.edge, transform=np.exp, title='Edge features'),
            'posterior/scatter/path': scatter_from_dicts('full', full_path_log_features,
                'estimate', log_features.path, transform=np.exp, title='Path features'),
            'posterior/scatter/markov_blanket': scatter_from_dicts('full', full_markov_log_features,
                'estimate', log_features.markov_blanket, transform=np.exp, title='Markov blanket features')
        })
        full_edge = list(full_edge_log_features.values())
        est_edge = list(log_features.edge.values())
        full_edge_ordered, est_edge_ordered = return_ordered_data(full_edge_log_features,
                log_features.edge, transform=np.exp)
        edge_corr = np.corrcoef(full_edge_ordered, est_edge_ordered)[0][1]

        full_path = list(full_path_log_features.values())
        est_path = list(log_features.path.values())
        full_path_ordered, est_path_ordered = return_ordered_data(full_path_log_features,
                log_features.path, transform=np.exp)
        path_corr = np.corrcoef(full_path_ordered, est_path_ordered)[0][1]

        full_markov_ordered, est_markov_ordered = return_ordered_data(full_markov_log_features,
                log_features.markov_blanket, transform=np.exp)

        markov_corr = np.corrcoef(full_markov_ordered, est_markov_ordered)[0][1]
        wandb.log({'edge correlation': edge_corr})
        wandb.log({'path correlation': path_corr})
        wandb.log({'markov correlation': markov_corr})

    # Save replay buffer
    with open(os.path.join(wandb.run.dir, 'replay_buffer.npz'), 'wb') as f:
        replay.save(f)
    wandb.save('replay_buffer.npz', policy='now')

if __name__ == '__main__':
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser('GFlowNet for Structure Learning')

    parser.add_argument('--num_variables', type=int, default=5,
        help='Number of variables (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=100,
        help='Number of samples (default: %(default)s)')
    parser.add_argument('--num_edges', type=int, default=5,
        help='Average number of parents (default: %(default)s)')
    parser.add_argument('--num_envs', type=int, default=8,
        help='Number of parallel environments (default: %(default)s)')
    parser.add_argument('--graph', type=str, default='erdos_renyi_lingauss',
        choices=['erdos_renyi_lingauss','erdos_renyi_lingauss_3_nodes', 'sachs_discrete', 'sachs_continuous', 'sachs_intervention', 'alarm'],
        help='Type of graph (default: %(default)s)')
    parser.add_argument('--num_samples_posterior', type=int, default=1000,
        help='Number of samples for the posterior estimate (default: %(default)s)')
    # Keeping the --herarchical option for backward compatibility, but unused (always set to True)
    parser.add_argument('--hierarchical', action='store_true',
        help='Use a hierarchical forward transition probability')
    parser.add_argument('--update_target_every', type=int, default=1000,
        help='Frequency of update for the target network (default: %(default)s)')
    parser.add_argument('--n_step', type=int, default=1,
        help='Maximum number of subsequences for multistep loss (default: %(default)s)')
    parser.add_argument('--prior', type=json.loads, default='{}',
        help='Arguments of the prior for the score.')
    parser.add_argument('--replay_capacity', type=int, default=100_000,
        help='Capacity of the replay buffer (default: %(default)s)')
    parser.add_argument('--replay_prioritized', action='store_true',
        help='Use Prioritized Experience Replay')
    parser.add_argument('--lr', type=float, default=1e-5,
        help='Learning rate (default: %(default)s)')
    parser.add_argument('--delta', type=float, default=1.,
        help='Value of delta for Huber loss (default: %(default)s)')

    parser.add_argument('--batch_size', type=int, default=32,
        help='Batch size (default: %(default)s)')
    parser.add_argument('--obs_start', type=int, default=5,
        help='startoff')
        
    parser.add_argument('--num_iterations', type=int, default=15,
        help='Number of iterations (default: %(default)s)')
    parser.add_argument('--prefill', type=int, default=1000,
        help='Number of iterations with a random policy to prefill '
             'the replay buffer (default: %(default)s)')
    parser.add_argument('--min_exploration', type=float, default=0.1,
        help='Minimum value of epsilon-exploration (default: %(default)s)')
    parser.add_argument('--log_every', type=int, default=50,
        help='Frequency for logging (default: %(default)s)')
    parser.add_argument('--start_to_increase_eps', type=float, default=0.5,
        help='the fraction of training iters to start increasing epsilon')
        
    
    parser.add_argument('--seed', type=int, default=0,
        help='Random seed (default: %(default)s)')
    parser.add_argument('--num_workers', type=int, default=4,
        help='Number of workers (default: %(default)s)')
    parser.add_argument('--mp_context', type=str, default='spawn',
        help='Multiprocessing context (default: %(default)s)')

    parser.add_argument('--vb', default=False, action='store_true',
                        help='use variational bayes setup to get parameter updates')
    parser.add_argument('--reset', default=False, action='store_true',
                        help='reset optimiser at each parameter update')
        
    parser.add_argument('--plot_vb_iter', default=False, action='store_true',
                        help='plot loss of gflownet update within one variational bayes mean update')

    parser.add_argument('--debug', default=False, action='store_true',
                        help='do not update means, just use true means for vb updates for debugging')
    parser.add_argument('--block_small_theta', default=False, action='store_true',
                        help='only sample graphs where theta values have magnitude larger than 0.5')

    parser.add_argument('--num_vb_updates', type=int, default=2000,
                        help='number of updates to gflownet per one update of parameters in VB setup')
    parser.add_argument('--introduce_intervention', type=int, default=2,
                        help='period in which one intervention is done. can be multiple nodes')
        
    parser.add_argument('--weight', type=float, default=0.5,
                        help='amount of weighting of KL term')
    parser.add_argument('--prop_obs', type=float, default=0.25,
                        help='proportion of obs data')
        
    parser.add_argument('--hetero_noise', default=False, action='store_true',
                        help='whether to have heterogeneious noise')
    
    parser.add_argument('--obs_noise', type=float, default=1.0,
                        help='likelihood variance in approximate posterior')
    parser.add_argument('--true_obs_noise', type=float, default=0.1,
                        help='true likelihood variance, data generated with this variance')
    parser.add_argument('--scale_edges', type=float, default=2.0,
                        help='upper limit for edge scale')
    parser.add_argument('--low_edges', type=float, default=0.5,
                        help='lower limit for edge scale')
    parser.add_argument('--int_nodes', nargs='+', type=int, default=None,
                        help='nodes to intervene on separated by space')
    parser.add_argument('--num_data_rounds', type=int, default=3,
                        help='number of rounds of introducing data')

    parser.add_argument('--prior_mean', type=int, default=0,
                        help='prior is a gaussian. Mean of that gaussian')
    parser.add_argument('--prior_var', type=int, default=1,
                        help='variance of gaussian prior')
    parser.add_argument('--full_cov', default=False, action='store_true')
    parser.add_argument('--random_init', default=False, action='store_true')
    parser.add_argument('--plot_cov', default=False, action='store_true')
    parser.add_argument('--graph_index', type=int, default=1,
                        help='indexing graphs for 3 node MEC check')
    parser.add_argument('--use_erdos_prior', default=False,
                        action='store_true',
                        help='whether to use erdos renyi prior over graphs')
    
    parser.add_argument('--keep_epsilon_constant', default=False,
                        action='store_true',
                        help='do not increase epsilon over time')

    parser.add_argument("--benchmarking", default=False, action='store_true',
                        help='use file saving locations for benchmarking')

    args = parser.parse_args()

    # Set has_interventional
    args.has_interventional = (args.graph == 'sachs_intervention')

    # Add information about the score
    if args.graph in ['erdos_renyi_lingauss', 'erdos_renyi_lingauss_3_nodes', 'sachs_continuous']:
        args.score = 'bge'
        default_prior = {'prior': 'uniform'}
    else:
        args.score = 'bdeu'
        default_prior = {
            'equivalent_sample_size': 10,
            'has_interventional': args.has_interventional,
            'prior': 'uniform',
            'beta': 0.1,
            'n_edges_per_node': 1
        }

    # Update prior parameters
    default_prior.update(args.prior)
    args.prior = default_prior
    
    # Set hierarchical to True
    args.hierarchical = True

    main(args)
