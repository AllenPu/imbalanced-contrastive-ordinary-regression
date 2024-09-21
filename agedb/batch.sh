#for i in 2 5 10 15 20 25 40 50; do
#    python train_rnc.py --soft_label --asymm --groups $i
#done
for i in 0.5 0.7 0.9 1 1.25 1.5 2 4; do
    python train_rnc.py --soft_label --symm --step $i
done
