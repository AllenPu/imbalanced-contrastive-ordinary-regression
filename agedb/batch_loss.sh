#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --oe
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --oe --pretrained
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --oe --weight_norm --pretrained
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --oe --norm --pretrained
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --oe --norm  --weight_norm --pretrained
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100 --pretrained
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100 --weight_norm --pretrained
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100 --norm --pretrained
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100 --norm  --weight_norm --pretrained
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --weight_norm
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --oe 
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --oe --weight_norm 
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --oe --norm 
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100  --oe --norm  --weight_norm 
#echo " train on the pure OE+MSE loss"
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100 --weight_norm 
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100 --norm 
#CUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output --epoch 100 --norm  --weight_norm 
#echo " train on the pure MSE"