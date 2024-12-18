#for i in 2 5 10 15 20 25 40 50; do
    #python train_rnc.py --ce --groups $i #tmux 0
#   python train_rnc.py --soft_label --asymm --groups $i #tmux 4
    #python train_rnc.py --soft_label --groups $i #tmux 5
#done

#for i in 0.5 0.7 0.9 1 1.25 1.5 2 4; do
#    python train_rnc.py --soft_label --step $i
#done

#python draw_tsne.py --model_name checkpoint/groups_10_lr_0.001_epoch_100_soft_label_asymm.pth --store_name soft_asymm
#python draw_tsne.py --model_name checkpoint/groups_10_lr_0.001_epoch_100_soft_label_symm.pth --store_name soft_symm
#python draw_tsne.py --model_name checkpoint/groups_10_lr_0.001_epoch_100_ce.pth --store_name ce
#python train_rnc.py --fine_tune --single
#python train_rnc.py --single
#python train_rnc.py --fine_tune --soft_label
#python train_rnc.py --soft_label
#python train_rnc.py --fine_tune  --soft_label --asymm
#python train_rnc.py --soft_label --asymm 

##### tmux 1

echo '===================================================================='
echo ' groups 10 fine_tune pre-trained epoch 100 asymm soft label'
CUDA_VISIBLE_DEVICES=1 python train_rnc.py --soft_label --asymm --groups 10 --epoch 100 --model_name '/home/rpu2/scratch/code/Rank-N-Contrast/save/AgeDB_models/RnC_AgeDB_resnet18_ep_400_lr_0.5_d_0.1_wd_0.0001_mmt_0.9_bsz_256_aug_crop,flip,color,grayscale_temp_2_label_l1_feature_l2_trial_0/ckpt_epoch_100.pth'
echo '===================================================================='
echo ' groups 10 fine_tune pre-trained epoch 150 asymm soft label'
CUDA_VISIBLE_DEVICES=1 python train_rnc.py --soft_label --asymm --groups 10 --epoch 100 --model_name '/home/rpu2/scratch/code/Rank-N-Contrast/save/AgeDB_models/RnC_AgeDB_resnet18_ep_400_lr_0.5_d_0.1_wd_0.0001_mmt_0.9_bsz_256_aug_crop,flip,color,grayscale_temp_2_label_l1_feature_l2_trial_0/ckpt_epoch_150.pth'
echo '===================================================================='
echo ' groups 10 fine_tune pre-trained epoch 200 asymm soft label'
CUDA_VISIBLE_DEVICES=1 python train_rnc.py --soft_label --asymm --groups 10 --epoch 100 --model_name '/home/rpu2/scratch/code/Rank-N-Contrast/save/AgeDB_models/RnC_AgeDB_resnet18_ep_400_lr_0.5_d_0.1_wd_0.0001_mmt_0.9_bsz_256_aug_crop,flip,color,grayscale_temp_2_label_l1_feature_l2_trial_0/ckpt_epoch_200.pth'
echo '===================================================================='
echo ' groups 20 fine_tune pre-trained epoch 100 asymm soft label'
CUDA_VISIBLE_DEVICES=1 python train_rnc.py --soft_label --asymm --groups 20 --epoch 100 --model_name '/home/rpu2/scratch/code/Rank-N-Contrast/save/AgeDB_models/RnC_AgeDB_resnet18_ep_400_lr_0.5_d_0.1_wd_0.0001_mmt_0.9_bsz_256_aug_crop,flip,color,grayscale_temp_2_label_l1_feature_l2_trial_0/ckpt_epoch_100.pth'
echo '===================================================================='
echo ' groups 20 fine_tune pre-trained epoch 150 asymm soft label'
CUDA_VISIBLE_DEVICES=1 python train_rnc.py --soft_label --asymm --groups 20 --epoch 100 --model_name '/home/rpu2/scratch/code/Rank-N-Contrast/save/AgeDB_models/RnC_AgeDB_resnet18_ep_400_lr_0.5_d_0.1_wd_0.0001_mmt_0.9_bsz_256_aug_crop,flip,color,grayscale_temp_2_label_l1_feature_l2_trial_0/ckpt_epoch_150.pth'
echo '===================================================================='
echo ' groups 20 fine_tune pre-trained epoch 200 asymm soft label'
CUDA_VISIBLE_DEVICES=1 python train_rnc.py --soft_label --asymm --groups 20 --epoch 100 --model_name '/home/rpu2/scratch/code/Rank-N-Contrast/save/AgeDB_models/RnC_AgeDB_resnet18_ep_400_lr_0.5_d_0.1_wd_0.0001_mmt_0.9_bsz_256_aug_crop,flip,color,grayscale_temp_2_label_l1_feature_l2_trial_0/ckpt_epoch_200.pth'