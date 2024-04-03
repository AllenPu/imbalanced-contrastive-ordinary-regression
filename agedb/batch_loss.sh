#CUDACUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --single_output
CUDACUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --ce
CUDACUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --soft_label
CUDACUDA_VISIBLE_DEVICES=2 python train_rnc_loss.py --la