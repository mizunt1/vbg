import sys, os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import matplotlib as mpl

from gflownet_sl.utils.metrics import return_file_paths
import seaborn as sns
import pandas as pd



# shd
methods = ["gflowdag", "dibs"]#, "mcmc_gibbs", "mcmc_mh", "bootstrap_ges", "bootstrap_pc"]
method_str = ["VBG", "DiBs"]#, "MCMC Gibbs", "MCMC MH", "BS GES", "BS PC"]
result = "results2"
file_path = return_file_paths(0, result, methods[0], base_dir="/network/scratch/m/mizu.nishikawa-toomey/vbg2/")["summary_all_data"]
metrics = list(pd.read_csv(file_path, index_col=0).columns.values)
print(metrics)

dfs = {}
for metric in metrics:
    data_for_metric = {}
    for method, stri in zip(methods, method_str):
        file_path = return_file_paths(0, result, method, base_dir="/network/scratch/m/mizu.nishikawa-toomey/vbg2/")["summary_all_data"]
        data = pd.read_csv(file_path, index_col=0)[metric]
        data_for_metric[stri] = data.to_list()
    dfs[metric] = pd.DataFrame.from_dict(data_for_metric, orient="index")
    
# sns.set(font="Verdana")
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

font = {'family' : 'serif',
        'size'   : 12}

mpl.rc('font', **font)


for metric in metrics:
    plt.figure(figsize=(6,5.7), dpi=100)
    plt.clf()
    print(metric)
    plot_is = sns.boxplot(data=(dfs[metric].T))
    plt.xticks(rotation=20)

    fig = plot_is.get_figure()
    
    fig.savefig("figs_vbg2/" + result + "_" + metric + ".png")
