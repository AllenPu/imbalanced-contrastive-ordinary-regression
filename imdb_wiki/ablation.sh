#CUDA_VISIBLE_DEVICE=0 python train_rnc.py --epoch 100 --soft_label --groups 10 --aug_model
CUDA_VISIBLE_DEVICES=1 python train_rnc.py --epoch 400 --lr 0.05 