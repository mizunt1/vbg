#!/bin/bash
#SBATCH --job-name=s5
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-6
WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt1
module load python/3.7
module load cuda/11.1/cudnn/8.0

source $HOME/python_envs/causal_proj/bin/activate
srun python main.py --num_samples 20000 --vb --num_iterations 10 --plot_vb_iter --num_vb_updates 1500 --log_every 1 --block_small_theta --full_cov  --seed 0 --random_init --num_variables 3 --graph erdos_renyi_lingauss_3_nodes --graph_index ${SLURM_ARRAY_TASK_ID}
