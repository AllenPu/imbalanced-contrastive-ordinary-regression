#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --account=def-boyuwang
#SBATCH --time=01-20:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mail-user=18651885620@163.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/home/ruizhipu/scratch/regression/imbalanced-contrastive-ordinary-regression//imdb_wiki/slurm_output/slurm-%j-%x.out
module load StdEnv/2020 cuda scipy-stack python/3.8
#
source /home/ruizhipu/envs/py38/bin/activate

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID


#python train.py --la --tau 0.5 --ranked_contra --lr $1  --groups $2 --epoch $3 --temp $4 
python train.py --lr 0.001 --groups 20 --epoch 150 --sigma 2 --soft_label True --ranked_contra True --output_file soft_0317_contra_