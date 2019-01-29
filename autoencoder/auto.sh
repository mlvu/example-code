#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C TitanX
#SBATCH --gres=gpu:1

module load cuda90/toolkit
module load cuDNN/cuda90rc  

source /home/pbloem/.bashrc

cd /home/pbloem/experiments/autoencoder

python /home/pbloem/git/machine-learning/code-samples/autoencoder/autoencoder.py -e 300 -l 0.005

wait          # wait until programs are finished

