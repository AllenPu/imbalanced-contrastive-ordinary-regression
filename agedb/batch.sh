for i in 2 5 10 15 20 25 40 50; do
    python train_rnc.py --soft_label --asymm --groups $i
done

