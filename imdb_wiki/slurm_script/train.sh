#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --account=def-boyuwang
#SBATCH --time=01-15:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mail-user=18651885620@163.com
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm_output/slurm-%j-%x.out
module load StdEnv/2020 cuda scipy-stack python/3.8
#
source /home/ruizhipu/envs/py38/bin/activate

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID


python train.py --data_dir /home/ruizhipu/scratch/regression/imbalanced-regression/imdb-wiki-dir/data --la True --ranked_contra True --tau 0.5 --lr $1  --groups $2 --epoch $3 --temp $4 