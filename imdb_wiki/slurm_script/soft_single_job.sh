#CUDA_VISIBLE_DEVICES=1 python train.py --lr 0.00001 --groups 20 --epoch 200 --temp 0.02 --batch_size 256 --diversity 1 --sigma 1 --ranked_contra --soft_label --ce --output_file softlabel_g20_1214 --data_dir /home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data
CUDA_VISIBLE_DEVICES=2 python train.py --lr 0.00001 --groups 20 --epoch 200 --temp 0.02 --batch_size 256 --diversity 1 --sigma 2 --ranked_contra --soft_label --ce --output_file softlabel_g20_1214 --data_dir /home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data