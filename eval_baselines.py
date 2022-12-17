import os
import subprocess
import pickle as pkl
import numpy as np
from numpy.random import default_rng
import jax
from jax import random
from gflownet_sl.utils.graph import sample_erdos_renyi_linear_gaussian, get_weighted_adjacency
from gflownet_sl.utils.sampling import sample_from_linear_gaussian
from gflownet_sl.metrics.metrics import expected_shd
from gflownet_sl.baselines.bootstrap.learners import GES, PC
from gflownet_sl.baselines.bootstrap.bootstrap import NonparametricDAGBootstrap
from gflownet_sl.baselines.gadget_beeps import BaselineGadget
from gflownet_sl.baselines.bcdnets.bcdnets import BCDNets  # fix: add to pythonpath, compile cython (see below)
# export PYTHONPATH=/Users/.../gflownet_sl:/Users/.../gflownet_sl/gflownet_sl/baselines/bcdnets
# in folder c_modules, run `cythonize -i mine.pyx` (note the extra initial line # distutils: language = c++)


def checkR():
    """check whether R version 4 is available"""
    bashcommand = "R --version"
    try:
        process = subprocess.Popen(bashcommand.split(), stdout=subprocess.PIPE)
    except FileNotFoundError:
        return False  # R is not installed
    output, error = process.communicate()
    has_r4 = '4' == output.split()[2].decode('UTF-8')[0]
    return has_r4


def generate_data(degree=1, dim=32, seed=0, path=None):
    rng = default_rng(seed)

    graph = sample_erdos_renyi_linear_gaussian(
        num_variables=dim,
        num_edges=degree * dim/2,
        loc_edges=0.0,
        scale_edges=1.0,
        obs_noise=0.1,
        rng=rng
    )
    data = sample_from_linear_gaussian(
        graph,
        num_samples=100,
        rng=rng
    )
    w_adjacency = get_weighted_adjacency(graph)
    if path is None:
        path = './dags_n{}d{}'.format(dim, degree)

    if not os.path.exists(path):
        os.mkdir(path)
    gtruth_path = os.path.join(path, 'ground_truth_W.pkl')
    if not os.path.exists(gtruth_path):
        pkl.dump(w_adjacency, open(gtruth_path, 'wb'))
    else:
        gt = pkl.load(open(gtruth_path, 'rb'))
        assert np.all(gt == w_adjacency), "there are stored results which don't match the newly generated DAG"

    return data, w_adjacency


def train_bcdnets(bcdnets_path):
    bcdnet = BCDNets(Xs=data.to_numpy(),
                     test_Xs=data.to_numpy(),
                     ground_truth_W=w_adjacency)
    bcdnets_posterior = bcdnet.best_train_posterior
    pkl.dump(bcdnets_posterior, open(bcdnets_path, 'wb'))
    return bcdnets_posterior


def eval_bcdnets_posterior(bcdnets_posterior):
    if bcdnets_posterior is not None:
        # clip
        threshold = .3
        bcdnets_posterior = np.where(np.abs(bcdnets_posterior) > threshold, bcdnets_posterior, 0)
        # binarize
        bcdnets_posterior = (bcdnets_posterior != 0).astype(float)
        print("Computing SHD for BCDNets:")
        print(expected_shd(bcdnets_posterior, binary_gtruth), flush=True)
    else:
        print("BCDNets posterior not available, skipping evaluation", flush=True)


def train_dagbootstrap(bootstrap_path, learner):
    seed = 1
    key = random.PRNGKey(seed)
    boot = NonparametricDAGBootstrap(
        learner=learner,
        verbose=False,
        n_restarts=20,
        no_bootstrap=False,
    )
    bootstrap_posterior = boot.sample_particles(
        key=key, n_samples=50, x=np.array(data),  # jnp.array(target.x),
        verbose_indication=False
    )
    pkl.dump(bootstrap_posterior, open(bootstrap_path, 'wb'))
    return bootstrap_posterior


def train_gadget(gadget_path):
    g = BaselineGadget(np.array(data), discrete=False)
    gadget_posterior, scores = g.sample()

    import sumu  # todo put this inside our gadget class
    gadget_posterior = np.array([sumu.bnet.family_sequence_to_adj_mat(dag) for dag in gadget_posterior])
    pkl.dump(gadget_posterior, open(gadget_path, 'wb'))
    return gadget_posterior


if __name__ == '__main__':
    r_available = checkR()
    degree = 4
    for dim in [20, 50]:
        print('starting dim=n{}, degree={}'.format(dim, degree), flush=True)
        path = './dags_n{}d{}'.format(dim, degree)
        data, w_adjacency = generate_data(degree, dim, path=path)
        binary_gtruth = (w_adjacency != 0).astype(float)

        bcdnets_path = os.path.join(path, 'dag_samples_bcdnets.pkl')
        if os.path.exists(bcdnets_path):
            print("found stored result for BCDNets", flush=True)
            bcdnets_posterior = pkl.load(open(bcdnets_path, 'rb'))
        elif jax.devices()[0].platform != "cpu":
            print("starting BCDNets train", flush=True)
            bcdnets_posterior = train_bcdnets(bcdnets_path)
        else:
            print("GPU not found, skipping BCDNets-training", flush=True)
            bcdnets_posterior = None
        eval_bcdnets_posterior(bcdnets_posterior)

        pcbootstrap_path = os.path.join(path, 'dag_samples_pc-bootstrap.pkl')
        if os.path.exists(pcbootstrap_path):
            print("found stored result for PCBootstrap", flush=True)
            pcbootstrap_posterior = pkl.load(open(pcbootstrap_path, 'rb'))
        elif r_available:
            print("starting PCBootstrap train", flush=True)
            pcbootstrap_posterior = train_dagbootstrap(pcbootstrap_path, PC())
        else:
            print("R version 4 not found, skipping pc-DAGbootstrap-training", flush=True)
            pcbootstrap_posterior = None
        if pcbootstrap_posterior is not None:
            print("Computing SHD for pc-DAGbootstrap:")
            print(expected_shd(pcbootstrap_posterior, binary_gtruth), flush=True)

        gesbootstrap_path = os.path.join(path, 'dag_samples_ges-bootstrap.pkl')
        if os.path.exists(gesbootstrap_path):
            print("found stored result for GESBootstrap", flush=True)
            gesbootstrap_posterior = pkl.load(open(gesbootstrap_path, 'rb'))
        elif r_available:
            print("starting GESBootstrap train", flush=True)
            gesbootstrap_posterior = train_dagbootstrap(gesbootstrap_path, GES())
        else:
            print("R version 4 not found, skipping ges-DAGbootstrap-training", flush=True)
            gesbootstrap_posterior = None
        if gesbootstrap_posterior is not None:
            print("Computing SHD for ges-DAGbootstrap:")
            print(expected_shd(gesbootstrap_posterior, binary_gtruth), flush=True)

        gadget_path = os.path.join(path, 'dag_samples_gadget.pkl')
        if os.path.exists(gadget_path):
            print("found stored result for Gadget", flush=True)
            gadget_posterior = pkl.load(open(gadget_path, 'rb'))
        elif r_available:
            print("starting Gadget train", flush=True)
            gadget_posterior = train_gadget(gadget_path)
        else:
            print("R version 4 not found, skipping ges-DAGbootstrap-training", flush=True)
            gadget_posterior = None
        if gadget_posterior is not None:
            print("Computing SHD for Gadget:")
            print(expected_shd(gadget_posterior, binary_gtruth), flush=True)
