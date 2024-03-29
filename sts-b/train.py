import os
import sys
import time
import random
import argparse
import shutil
import logging
import torch
import numpy as np
from allennlp.data.iterators import BasicIterator

from preprocess import build_tasks
from models import build_model
from trainer import build_trainer
from evaluate import evaluate
from util import device_mapping, query_yes_no, resume_checkpoint

def main(arguments):
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--cuda', help='-1 if no CUDA, else gpu id (single gpu is enough)', type=int, default=0)
    parser.add_argument('--random_seed', help='random seed to use', type=int, default=111)

    # Paths and logging
    parser.add_argument('--log_file', help='file to log to', type=str, default='_training.log')
    parser.add_argument('--store_root', help='store root path', type=str, default='checkpoint')
    parser.add_argument('--store_name', help='store name prefix for current experiment', type=str, default='sts')
    parser.add_argument('--suffix', help='store name suffix for current experiment', type=str, default='')
    parser.add_argument('--word_embs_file', help='file containing word embs', \
                        type=str, default='//home/rpu2/scratch/data/imbalanced-regression/sts-b-dir/glove/glove.840B.300d.txt')
    # compute canada :  '/home/ruizhipu/scratch/regression/imbalanced-regression/sts-b-dir/glove/glove.840B.300d.txt'

    # Training resuming flag
    parser.add_argument('--resume', help='whether to resume training', action='store_true', default=False)

    # Tasks
    parser.add_argument('--task', help='training and evaluation task', type=str, default='sts-b')

    # Preprocessing options
    parser.add_argument('--max_seq_len', help='max sequence length', type=int, default=40)
    parser.add_argument('--max_word_v_size', help='max word vocab size', type=int, default=30000)

    # Embedding options
    parser.add_argument('--dropout_embs', help='dropout rate for embeddings', type=float, default=.2)
    parser.add_argument('--d_word', help='dimension of word embeddings', type=int, default=300)
    parser.add_argument('--glove', help='1 if use glove, else from scratch', type=int, default=1)
    parser.add_argument('--train_words', help='1 if make word embs trainable', type=int, default=0)

    # Model options
    parser.add_argument('--d_hid', help='hidden dimension size', type=int, default=1500)
    parser.add_argument('--n_layers_enc', help='number of RNN layers', type=int, default=2)
    parser.add_argument('--n_layers_highway', help='number of highway layers', type=int, default=0)
    parser.add_argument('--dropout', help='dropout rate to use in training', type=float, default=0.2)

    # Training options
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('--optimizer', help='optimizer to use', type=str, default='adam')
    parser.add_argument('--lr', help='starting learning rate', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'])
    parser.add_argument('--huber_beta', type=float, default=0.3, help='beta for huber loss')
    parser.add_argument('--max_grad_norm', help='max grad norm', type=float, default=5.)
    parser.add_argument('--val_interval', help='number of iterations between validation checks', type=int, default=400)
    parser.add_argument('--max_vals', help='maximum number of validation checks', type=int, default=100)
    #
    parser.add_argument('--patience', help='patience for early stopping', type=int, default=10)

    # imbalanced related
    # LDS
    parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
    parser.add_argument('--lds_kernel', type=str, default='gaussian',
                        choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
    parser.add_argument('--lds_ks', type=int, default=5, help='LDS kernel size: should be odd number')
    parser.add_argument('--lds_sigma', type=float, default=2, help='LDS gaussian/laplace kernel sigma')
    # FDS
    parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
    parser.add_argument('--fds_kernel', type=str, default='gaussian',
                        choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
    parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
    parser.add_argument('--fds_sigma', type=float, default=2, help='FDS gaussian/laplace kernel sigma')
    parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
    parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
    parser.add_argument('--bucket_num', type=int, default=50, help='maximum bucket considered for FDS')
    parser.add_argument('--bucket_start', type=int, default=0, help='minimum(starting) bucket for FDS')
    parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

    # re-weighting: SQRT_INV / INV
    parser.add_argument('--reweight', type=str, default='none', choices=['none', 'sqrt_inv', 'inverse'],
                        help='cost-sensitive reweighting scheme')
    # two-stage training: RRT
    parser.add_argument('--retrain_fc', action='store_true', default=False,
                        help='whether to retrain last regression layer (regressor)')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained checkpoint file path to load backbone weights for RRT')
    # evaluate only
    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate only flag')
    parser.add_argument('--eval_model', type=str, default='', help='the model to evaluate on; if not specified, '
                                                                   'use the default best model in store_dir')
    # group-wise
    parser.add_argument('--total_groups', type=float, default=5)
    parser.add_argument('--group_wise', action='store_true')
    parser.add_argument("--ranked_contra", action='store_true', help='use contastive loss or not')
    parser.add_argument('--groups', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--g_dis', action='store_true')
    parser.add_argument('--la', action='store_true')
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--patience_epoch', type=int, default=400)
    parser.add_argument('--tsne', type=bool, default=False)
    parser.add_argument('--temp', type=float, default=0.07)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--soft_label', action='store_true')
    parser.add_argument('--ce', action='store_false',  help='if use the cross_entropy /la or not')
    parser.add_argument('--scale', type=float, default=1, help='scale of the sharpness in soft label')
    parser.add_argument('--output_file', default='./results_', help='the output directory')
    parser.add_argument('--encoder_only', type= bool, default=False, help='Only train encoder')


    args = parser.parse_args(arguments)

    os.makedirs(args.store_root, exist_ok=True)

    if not args.lds and args.reweight != 'none':
        args.store_name += f'_{args.reweight}'
    if args.lds:
        args.store_name += f'_lds_{args.lds_kernel[:3]}_{args.lds_ks}'
        if args.lds_kernel in ['gaussian', 'laplace']:
            args.store_name += f'_{args.lds_sigma}'
    if args.fds:
        args.store_name += f'_fds_{args.fds_kernel[:3]}_{args.fds_ks}'
        if args.fds_kernel in ['gaussian', 'laplace']:
            args.store_name += f'_{args.fds_sigma}'
        args.store_name += f'_{args.start_update}_{args.start_smooth}_{args.fds_mmt}'
    if args.retrain_fc:
        args.store_name += f'_retrain_fc'

    if args.loss == 'huber':
        args.store_name += f'_{args.loss}_beta_{args.huber_beta}'
    else:
        args.store_name += f'_{args.loss}'

    args.store_name += f'_seed_{args.random_seed}_valint_{args.val_interval}_patience_epoch_{args.patience_epoch}' \
                       f'_opt_{args.optimizer}_lr_{args.lr}_bs_{args.batch_size}_grouwise_{args.group_wise}_groups_{args.groups}' \
                       f'_sigma_{args.sigma}_epoch_{args.epoch}'
    args.store_name += f'_{args.suffix}' if len(args.suffix) else ''

    args.store_dir = os.path.join(args.store_root, args.store_name)

    if not args.evaluate and not args.resume:
        if os.path.exists(args.store_dir):
            shutil.rmtree(args.store_dir)
            print(args.store_dir + ' removed.\n')
            '''
            if query_yes_no('overwrite previous folder: {} ?'.format(args.store_dir)):
                shutil.rmtree(args.store_dir)
                print(args.store_dir + ' removed.\n')
            else:
                raise RuntimeError('Output folder {} already exists'.format(args.store_dir))
            '''
        else:
            logging.info(f"===> Creating folder: {args.store_dir}")
        #if not os.path.exists(args.store_dir):
            os.makedirs(args.store_dir)

    # Logistics
    logging.root.handlers = []
    if os.path.exists(args.store_dir):
        #
        log_file = 'Soft_label_lr_' + str(args.lr) + '_groups_' + str(args.groups) + '_sigma_' + str(args.sigma) + \
            '_temp_' + str(args.temp) + '_patience_epoch_' + str(args.patience_epoch) + args.log_file
        #
        log_file = os.path.join(args.store_dir, log_file)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ])
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[logging.StreamHandler()]
        )
    logging.info(args)

    seed = random.randint(1, 10000) if args.random_seed < 0 else args.random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda >= 0:
        logging.info("Using GPU %d", args.cuda)
        torch.cuda.set_device(args.cuda)
        torch.cuda.manual_seed_all(seed)
    logging.info("Using random seed %d", seed)

    # Load tasks
    logging.info("Loading tasks...")
    start_time = time.time()
    tasks, vocab, word_embs = build_tasks(args)
    logging.info('\tFinished loading tasks in %.3fs', time.time() - start_time)

    # Build model
    logging.info('Building model...')
    start_time = time.time()
    model = build_model(args, vocab, word_embs, tasks)
    print(model)
    logging.info('\tFinished building model in %.3fs', time.time() - start_time)

    # Set up trainer
    iterator = BasicIterator(args.batch_size)
    trainer, train_params, opt_params = build_trainer(args, model, iterator)
    #print('tasks', tasks)
    # Train
    #
    if tasks and not args.evaluate:
        if args.retrain_fc and len(args.pretrained):
            model_path = args.pretrained
            assert os.path.isfile(model_path), f"No checkpoint found at '{model_path}'"
            model_state = torch.load(model_path, map_location=device_mapping(args.cuda))
            trainer._model = resume_checkpoint(trainer._model, model_state, backbone_only=True)
            logging.info(f'Pre-trained backbone weights loaded: {model_path}')
            logging.info('Retrain last regression layer only!')
            for name, param in trainer._model.named_parameters():
                if "sts-b_pred_layer" not in name:
                    param.requires_grad = False
            logging.info(f'Only optimize parameters: {[n for n, p in trainer._model.named_parameters() if p.requires_grad]}')
            to_train = [(n, p) for n, p in trainer._model.named_parameters() if p.requires_grad]
        elif args.encoder_only:
            to_train = [(n, p) for n, p in trainer._model.pair_encoder.named_parameters() if p.requires_grad]
        else:
            to_train = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        trainer.train(tasks, args.val_interval, to_train, opt_params, args.resume)
    else:
        logging.info("Skipping training...")

    logging.info('Testing on test set...')
    model_path = os.path.join(args.store_dir, "model_state_best.th") if not len(args.eval_model) else args.eval_model
    assert os.path.isfile(model_path), f"No checkpoint found at '{model_path}'"
    logging.info(f'Evaluating {model_path}...')
    model_state = torch.load(model_path, map_location=device_mapping(args.cuda))
    model = resume_checkpoint(model, model_state, args.group_wise, args.groups)
    te_preds, te_labels, _ = evaluate(
        model, tasks, iterator, cuda_device=args.cuda, split="test", store_name=args.store_name, ranked_contra=args.ranked_contra)
    if not len(args.eval_model):
        np.savez_compressed(os.path.join(args.store_dir, f"{args.store_name}.npz"), preds=te_preds, labels=te_labels)

    logging.info("Done testing.")

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
