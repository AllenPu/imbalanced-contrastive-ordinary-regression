CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --oe
CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --oe --weight_norm
CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --oe --norm
CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --oe --norm  --weight_norm
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --weight_norm
#CUDACUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --ce --epoch 150 --lr 5e-4
#CUDACUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --la --epoch 150
#CUDACUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --soft_label --epoch 150##