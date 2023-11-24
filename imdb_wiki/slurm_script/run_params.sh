#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --account=def-boyuwang
#SBATCH --time=02-15:00
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
python train.py --lr 0.001 --groups 10 --epoch 250 --temp 0.05 --sigma 1 --ranked_contra --soft_label --ce --scale 0.95  --diversity 0.5 --output_file softlabel_s_0.95d0.5__1123_