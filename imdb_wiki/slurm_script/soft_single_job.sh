CUDA_VISIBLE_DEVICES=2 python train.py --lr 0.0001 --sigma 1 --epoch 200 --soft_label  --ce --groups 20 --ranked_contra --temp 0.07 --output_file soft_imdb_1105_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data