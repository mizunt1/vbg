import wandb
import pandas as pd
from collections import defaultdict
import json
# this script scrapes a project (containing one inference method), 
# loops through all the seeds and saves all the metrics for each seed for that one method.
# means and sd for one metric not calculated, this is done automatically by the plotting script
entity, project = "mizunt", "int_exps" 
api = wandb.Api(timeout=19)
runs = api.runs(entity + "/" + project)

def get_runs(method, num_var):
    return api.runs('mizunt/int_exps',
        filters={'$and': [{
            'state': {'$eq': 'finished'}
            #'int_nodes': {'$eq': [1]},
        }]}
    )

def combine_runs(methods, base_dir, nodes_5, num_var):
    for method in methods:
        project = method
        runs = get_runs(project, num_var)

        edge_mse = defaultdict()
        path_mse = defaultdict()
        markov_mse = defaultdict()
        shd = defaultdict()
        auroc = defaultdict()
        nll = defaultdict()
        mse = defaultdict()
        for run_ in runs:
            summary = run_.summary
            seed = str(json.loads(run_.json_config)['seed']['value'])
            if nodes_5:
                edge_mse[str(seed)] = summary['edge mse']
                path_mse[str(seed)] = summary['path mse']
                markov_mse[str(seed)] = summary['markov mse']
            shd[str(seed)] = summary['metrics/shd/mean']
            auroc[str(seed)] = summary['metrics/thresholds']['roc_auc']
            nll[str(seed)] = summary['negative log like']
            mse[str(seed)] = summary['mse of mean']
        if nodes_5:
            df = pd.DataFrame(
                list(zip(edge_mse.values(),
                         path_mse.values(),
                         markov_mse.values(), shd.values(),
                         auroc.values(), nll.values(),
                         mse.values())),
                columns=['edge mse', 'path mse', 'markov mse', 'shd', 'auroc', 'nll', 'mse theta'])
        else:
            df = pd.DataFrame(
                list(zip(shd.values(),
                         auroc.values(), nll.values(),
                         mse.values())),
                columns=['shd', 'auroc', 'nll', 'mse theta'])
        df.to_csv(base_dir +'/' + project + '.csv')
        return df
        

if __name__ == '__main__':
    nodes_5 = True
    num_var = 5
    if nodes_5:
        base_dir = '/network/scratch/m/mizu.nishikawa-toomey/sl/n20'
        projects = ['int_exps']
    else:
        methods = ['ges_arxiv2_n5', 'bcd_arxiv2', 'vbg_arxiv3_n5', 'dibs_plus_arxiv2_n5', 'pc_arxiv2_n5', 'dibs_arxiv2_n5', 'gibs_arxiv2_n5',  'mh_arxiv2_n5', 'vbg_arxiv2_w_0.5']
        methods = ['mh_arxiv2_n5_burn', 'mh_arxiv2_n20_theta']

    df = combine_runs(projects, base_dir, nodes_5, num_var)
    import pdb
    pdb.set_trace()
