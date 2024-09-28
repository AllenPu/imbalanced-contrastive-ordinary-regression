#for e in 50 70 90; do
    #jobs='lr'_${i}_'tau_0.5'_'group'_${g}_'epoch'_${e}_'temp'_${temp}
    #echo ${jobs}
#    python train_rnc.py --model_name ./checkpoint/Supervised_group_lr_0.001_epoch_300_bs_128_groups_10.pth --epoch $e --soft_label --asymm
    #sbatch --job-name=${jobs} ./slurm_jobs/train.sh ${i} 
#done
#python train_rnc.py --model_name ./checkpoint/Supervised_group_lr_0.001_epoch_300_bs_128_groups_10.pth --lr 5e-8 --encoder_lr 1e-10 --soft_label --asymm
python train_rnc.py --model_name ./checkpoint/Supervised_group_lr_0.001_epoch_300_bs_128_groups_10.pth --lr 5e-9 --encoder_lr 1e-10 --soft_label --asymm
echo ' linear lr 5e-9 encoder 1e-10'