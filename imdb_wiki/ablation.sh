#python train_rnc.py --epoch 30 --soft_label
#python train_rnc.py --epoch 40 --soft_label
#python train_rnc.py --epoch 50 --soft_label
#python train_rnc.py --epoch 60 --soft_label
#python train_rnc.py --epoch 40 --soft_label --lr 0.001
#python train_rnc.py --epoch 40 --soft_label --lr 0.005
#python train_rnc.py --epoch 40 --soft_label --lr 0.001 --groups 20
#python train_rnc.py --epoch 40 --soft_label --lr 0.001 --groups 25
#python train_rnc.py --epoch 40 --soft_label --lr 0.005 --groups 20
#python train_rnc.py --epoch 40 --soft_label --lr 0.005 --groups 25
echo 'aug'
python train_rnc.py --epoch 30 --soft_label --aug_model
python train_rnc.py --epoch 40 --soft_label --aug_model
python train_rnc.py --epoch 50 --soft_label --aug_model
python train_rnc.py --epoch 60 --soft_label --aug_model
python train_rnc.py --epoch 40 --soft_label --lr 0.001 --aug_model
python train_rnc.py --epoch 40 --soft_label --lr 0.005 --aug_model
python train_rnc.py --epoch 40 --soft_label --lr 0.001 --groups 20 --aug_model
python train_rnc.py --epoch 40 --soft_label --lr 0.001 --groups 25 --aug_model
python train_rnc.py --epoch 40 --soft_label --lr 0.005 --groups 20 --aug_model
python train_rnc.py --epoch 40 --soft_label --lr 0.005 --groups 25 --aug_model