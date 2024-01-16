#CUDA_VISIBLE_DEVICES=1 python train.py --lr 0.001 --groups 20 --epoch 200 --temp 0.02 --sigma 1 --ranked_contra --soft_label --ce  --batch_size 64 --fd_ratio 1 --output_file softlabel_z_diver_1209_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
#CUDA_VISIBLE_DEVICES=1 python train.py --lr 0.002 --groups 20 --epoch 200 --temp 0.02 --sigma 1 --ranked_contra --soft_label --ce  --batch_size 64 --fd_ratio 1 --output_file softlabel_z_diver_1209_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
#CUDA_VISIBLE_DEVICES=1 python train.py --lr 0.0005 --groups 20 --epoch 200 --temp 0.02 --sigma 1 --ranked_contra --soft_label --ce  --batch_size 64 --fd_ratio 1 --output_file softlabel_z_diver_1209_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
#CUDA_VISIBLE_DEVICES=2 python train.py --lr 0.0001 --batch_size 64 --groups 20 --epoch 200 --temp 0.02 --sigma 1 --ranked_contra --soft_label --ce --fd_ratio 1 --output_file softlabel_z_diver_1209_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
#CUDA_VISIBLE_DEVICES=2 python train.py --lr 0.001 --batch_size 64 --groups 25 --epoch 200 --temp 0.02 --sigma 1 --ranked_contra --soft_label --ce --fd_ratio 1 --output_file softlabel_z_diver_1209_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
#CUDA_VISIBLE_DEVICES=2 python train.py --lr 0.001 --batch_size 64 --groups 20 --epoch 200 --temp 0.02 --sigma 1 --ranked_contra --soft_label --ce --fd_ratio 2 --output_file softlabel_z_diver_1209_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
#CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --batch_size 64 --groups 25 --epoch 300 --temp 0.02 --sigma 1 --ranked_contra --soft_label --ce --fd_ratio 2 --output_file softlabel_z_diver_1211_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
#CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --batch_size 64 --groups 25 --epoch 300 --temp 0.05 --sigma 1 --ranked_contra --soft_label --ce --fd_ratio 2 --output_file softlabel_z_diver_1211_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
#CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.002 --batch_size 64 --groups 25 --epoch 250 --temp 0.05 --sigma 1 --ranked_contra --soft_label --ce --fd_ratio 3 --output_file softlabel_z_diver_1211_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
#CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.0015 --batch_size 256 --groups 20 --epoch 300 --temp 0.02 --sigma 2 --ranked_contra --soft_label --ce --fd_ratio 2 --output_file softlabel_z_diver_1214_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
#CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.0015 --batch_size 256 --groups 20 --epoch 300 --temp 0.07 --sigma 2 --ranked_contra --soft_label --ce --fd_ratio 2 --output_file softlabel_z_diver_1214_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
#CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.0051 --batch_size 256 --groups 20 --epoch 300 --temp 0.1 --sigma 2 --ranked_contra --soft_label --ce --fd_ratio 2 --output_file softlabel_z_diver_1214_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
#CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --batch_size 256 --groups 25 --epoch 300 --temp 0.07 --sigma 0.5 --ranked_contra --soft_label --ce --fd_ratio 2 --output_file softlabel_z_diver_1216_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
#CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --batch_size 256 --groups 25 --epoch 300 --temp 0.07 --sigma 0.1 --ranked_contra --soft_label --ce --fd_ratio 2 --output_file softlabel_z_diver_1216_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
#CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --batch_size 256 --groups 25 --epoch 300 --temp 0.1 --sigma 0.5 --ranked_contra --soft_label --ce --fd_ratio 2 --output_file softlabel_z_diver_1216_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data
CUDA_VISIBLE_DEVICES=2 python train.py --lr 0.001 --groups 5 --epoch 200 --temp 0.05 --sigma 1 --ranked_contra --soft_label  --fd_ratio 1 --output_file softlabel_icml_ --data_dir /home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data