for i in 2 5 10 15 20 25 40 50; do
    echo 'the group is ' $i
    echo '----------------------'
    python train_rnc.py --groups $i --epoch 20 --soft_label
done
for i in 2 5 10 15 20 25 40 50; do
    echo 'the group is ' $i
    echo '----------------------'
    python train_rnc.py --groups $i --epoch 20 --ce
done
for i in 2 5 10 15 20 25 40 50; do
    echo 'the group is ' $i
    echo '----------------------'
    python train_rnc.py --groups $i --epoch 20 --la
done