for i in 2 5 10 15 20 25 40 50; do
    python train_rnc.py --ce --groups $i
    #python train_rnc.py --soft_label --asymm --groups $i
    #python train_rnc.py --soft_label --groups $i
done

#for i in 0.5 0.7 0.9 1 1.25 1.5 2 4; do
#    python train_rnc.py --soft_label --step $i
#done

#python draw_tsne.py --model_name checkpoint/groups_10_lr_0.001_epoch_100_soft_label_asymm.pth --store_name soft_asymm
#python draw_tsne.py --model_name checkpoint/groups_10_lr_0.001_epoch_100_soft_label_symm.pth --store_name soft_symm
#python draw_tsne.py --model_name checkpoint/groups_10_lr_0.001_epoch_100_ce.pth --store_name ce
