#!/bin/bash
#SBATCH --job-name=s5
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-25%15

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt1
module load python/3.7
module load cuda/11.1/cudnn/8.0

source $HOME/python_envs/causal_proj/bin/activate
python main.py --num_samples 500 --vb --num_iterations 15 --plot_vb_iter --num_vb_updates 2000 --log_every 1 --block_small_theta --start_to_increase_eps 0.10 --random_init --seed $SLURM_ARRAY_TASK_ID --true_obs_noise 0.1 --obs_noise 0.1 --full_cov --plot_cov --weight 0.1
