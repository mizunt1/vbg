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
                    filters={
                        "config.num_variables": 5})

def combine_runs(methods, base_dir, nodes_5, num_var):
    for method in methods:
        project = method
        runs = get_runs(project, num_var)

        shd_obs = defaultdict()
        auroc_obs = defaultdict()
        nll_obs = defaultdict()
        mse_obs = defaultdict()
        shd_int = defaultdict()
        auroc_int = defaultdict()
        nll_int = defaultdict()
        mse_int = defaultdict()

        for run_ in runs:
            seed = str(json.loads(run_.json_config)['seed']['value'])
            summary = run_.summary
            if json.loads(run_.json_config)['int_nodes']['value'] == None:
                shd_obs[str(seed)] = summary['metrics/shd/mean']
                auroc_obs[str(seed)] = summary['metrics/thresholds']['roc_auc']
                nll_obs[str(seed)] = summary['negative log like']
                mse_obs[str(seed)] = summary['mse of mean']

            else:
                shd_int[str(seed)] = summary['metrics/shd/mean']
                auroc_int[str(seed)] = summary['metrics/thresholds']['roc_auc']
                nll_int[str(seed)] = summary['negative log like']
                mse_int[str(seed)] = summary['mse of mean']

        df_obs = pd.DataFrame(
                list(zip(shd_obs.values(),
                         auroc_obs.values(), nll_obs.values(),
                         mse_obs.values())),
                columns=['shd', 'auroc', 'nll', 'mse theta'])
        
        df_int = pd.DataFrame(
            list(zip(shd_int.values(),
                     auroc_int.values(), nll_int.values(),
                     mse_int.values())),
            columns=['shd', 'auroc', 'nll', 'mse theta'])

        df_int.to_csv(base_dir +'/' + project +  'int.csv')
        df_obs.to_csv(base_dir +'/' + project + 'obs.csv')
        return df_obs, df_int

if __name__ == '__main__':
    nodes_5 = True
    num_var = 5
    if nodes_5:
        base_dir = 'int_exps/'
        projects = ['int_exps']
    else:
        methods = ['ges_arxiv2_n5', 'bcd_arxiv2', 'vbg_arxiv3_n5', 'dibs_plus_arxiv2_n5', 'pc_arxiv2_n5', 'dibs_arxiv2_n5', 'gibs_arxiv2_n5',  'mh_arxiv2_n5', 'vbg_arxiv2_w_0.5']
        methods = ['mh_arxiv2_n5_burn', 'mh_arxiv2_n20_theta']

    df_obs, df_int = combine_runs(projects, base_dir, nodes_5, num_var)
    
