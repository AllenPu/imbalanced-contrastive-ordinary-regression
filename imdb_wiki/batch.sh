for e in 50 70 90; do
    #jobs='lr'_${i}_'tau_0.5'_'group'_${g}_'epoch'_${e}_'temp'_${temp}
    #echo ${jobs}
    python train_rnc.py --model_name ./checkpoint/Supervised_group_lr_0.001_epoch_300_bs_128_groups_10.pth --epoch $e
    #sbatch --job-name=${jobs} ./slurm_jobs/train.sh ${i} 
done
