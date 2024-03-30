for i in 2 5 10 15 20 25 40 50; do
    python train_rnc.py --groups $i --epoch 20 --soft_label
done
