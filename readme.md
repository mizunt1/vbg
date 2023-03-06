# Variational Bayes GFlowNet
Extension to the DAG-GFlowNet which finds edge weights of nodes by assuming a linear-Gaussian relationship between nodes.

## Executing the program
The following line of code with generate synthetic data for 5 nodes, then infer the graph and the edge weights of the linear Gaussian relationships. Results are logged on wandb.
```
main.py --num_samples 500 --vb --num_iterations 15 --plot_vb_iter --num_vb_updates 2000 --log_every 1 --block_small_theta --start_to_increase_eps 0.10 --random_init --true_obs_noise 0.1 --obs_noise 0.1 --full_cov --plot_cov --weight 0.1
``` 


