#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --account=def-boyuwang
#SBATCH --time=02-10:00
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


#python train.py --la --tau 0.5 --lr 0.0002 --sigma 1 --epoch 200 --soft_label
# --ce means no standard ce but soft label ce
python train.py --lr $1 --groups $2 --epoch $3 --sigma $4 --soft_label --ce --output_file soft_batchrun_1031_no_contra_
#python train.py --lr $1 --groups $2 --epoch $3 --ranked_contra --temp $4 --sigma $5 --soft_label --ce --output_file soft_batchrun_