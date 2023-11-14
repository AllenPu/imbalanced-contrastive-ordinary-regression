#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --account=def-boyuwang
#SBATCH --time=01-00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mail-user=18651885620@163.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/home/ruizhipu/scratch/regression/imbalanced-contrastive-ordinary-regression//agedb/slurm_output/slurm-%j-%x.out
module load StdEnv/2020 cuda scipy-stack python/3.8
#
source /home/ruizhipu/envs/py38/bin/activate

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID


#python train.py --la --tau 0.5 --lr 0.0002 --temp 0.03 --sigma 1 --epoch 200 --ranked_contra --soft_label --output_file softlabel_
#python train.py --lr 0.001 --groups 10 --epoch 250 --temp 0.05 --sigma 1 --ranked_contra --soft_label --ce --scale 1.2 --output_file softlabel_scale1.2_1109_
python train.py --lr 0.0015 --groups 20 --epoch 250 --temp 0.02 --sigma 2 --ranked_contra --soft_label --ce --diversity 0.5 --output_file softlabel_diversity0.5_1113_
#python train.py --lr 0.001 --groups 10 --epoch 250 --temp 0.05 --sigma 1 --ranked_contra --soft_label --ce --scale 0.7 --output_file softlabel_scale0.7_1103_