#CUDA_VISIBLE_DEVICES=1 python train.py --lr 0.0001 --groups 20 --epoch 200 --temp 0.05 --batch_size 256 --diversity 2 --sigma 0.5 --ranked_contra --soft_label --output_file softlabel_g20_1214 --data_dir /home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data
#CUDA_VISIBLE_DEVICES=2 python train.py --lr 0.00005 --groups 20 --epoch 200 --temp 0.07 --batch_size 256 --diversity 2 --sigma 0.1 --ranked_contra --soft_label --output_file softlabel_g20_1214 --data_dir /home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data
#CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.0001 --groups 20 --epoch 200 --batch_size 256 --more_train True --ranked_contra --output_file softlabel_g20_1225 --data_dir /home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data
#CUDA_VISIBLE_DEVICES=1 python train.py --lr 0.001 --groups 20 --epoch 300 --batch_size 256 --lds True --smooth True --output_file smooth_1229_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data
#CUDA_VISIBLE_DEVICES=2 python train.py --lr 0.0001 --group_mode b_g --groups 20 --epoch 200 --batch_size 256 --lds True --ce --output_file smooth_eq_1225 --data_dir /home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data
#CUDA_VISIBLE_DEVICES=1 python train.py --lr 0.0005 --groups 20 --epoch 300 --group_mode b_g --batch_size 256 --ranked_contra --ce --lds True --output_file bg_g20_1231 --data_dir /home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data
CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.0001 --groups 20 --epoch 200 --aug --temp 2 --soft_label --output_file aug_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data
#CUDA_VISIBLE_DEVICES=1 python train.py --lr 0.0001 --groups 5 --epoch 250 --ranked_contra --soft_label --diversity 1 --output_file softlabel_g5_0113 --data_dir /home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data
