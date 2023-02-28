# Variational Bayes GFlowNet
Extension to the DAG-GFlowNet which finds edge weights of nodes by assuming a linear-Gaussian relationship between nodes.

## Executing the program
The following line of code with generate synthetic data for 5 nodes, then infer the graph and the edge weights of the linear Gaussian relationships. Results are logged on wandb.
```
python main.py --num_samples 500 --vb --num_iterations 15 --plot_vb_iter --num_vb_updates 2000 --log_eve
ry 1 --block_small_theta --full_cov --random_init
``` 


