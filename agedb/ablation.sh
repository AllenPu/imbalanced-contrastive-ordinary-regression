#python train_rnc.py --epoch 50 --ce --groups 20 --epoch 50
#python train_rnc.py --epoch 50 --soft_label --groups 20 --epoch 50
#python train_rnc.py --epoch 50 --la --groups 20 --epoch 50
CUDACUDA_VISIBLE_DEVICES=2 python train_rnc.py --soft_label --epoch 150 --norm