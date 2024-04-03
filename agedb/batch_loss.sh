#CUDACUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output
CUDACUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --ce --epoch 150
CUDACUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --la --epoch 150
CUDACUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --soft_label --epoch 150