for i in 2 5 10 15 20 25 40 50; do
    for j in 40, 50, 70; do
        echo 'the group is ' $i 'the rpoch is ' $j
        echo '----------------------'
        CUDA_VISIBLE_DEVICES=0 python train_rnc.py --groups $i --epoch $j --soft_label
        ptrhon draw_tsne.py --groups $i --epoch $j
    done
done

