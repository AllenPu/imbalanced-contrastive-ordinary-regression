#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --account=def-boyuwang
#SBATCH --time=00-04:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mail-user=18651885620@163.com
#SBATCH --mail-type=FAIL
#SBATCH --output=/home/ruizhipu/scratch/regression/imbalanced-contrastive-ordinary-regression/sts-b/slurm_output/slurm-%j-%x.out
module load StdEnv/2020 cuda scipy-stack python/3.7
#
source /home/ruizhipu/sts/bin/activate

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "current lr status : " $1
echo "current groups status : " $2

python train.py --group_wise --ranked_contra --la --lr $1  --groups $2 --sigma $3 --epoch $4 --temp $5