import logging
import tqdm
import numpy as np
import torch




def extract_feature(model, tasks, iterator, cuda_device, args, split="val", tsne=True):
    '''Evaluate on a dataset'''
    model.eval()

    all_preds = {}
    n_overall_examples = 0
    for task in tasks:
        n_examples = 0
        task_preds, task_idxs, task_labels = [], [], []
        group_pred, group_gt, embed = [], [], []
        if split == "val":
            dataset = task.val_data
        elif split == 'train':
            dataset = task.train_data
        elif split == "test":
            dataset = task.test_data
        generator = iterator(dataset, num_epochs=1, shuffle=False)
        generator_tqdm = tqdm.tqdm(
            generator, total=iterator.get_num_batches(dataset), disable=True)
        for batch in generator_tqdm:
            tensor_batch = batch
            out = model.forward(task, **tensor_batch)
            n_examples += batch['label'].size()[0]
            preds, _ = out['logits'].max(dim=1)
            #task_preds += list(preds.data.cpu().numpy())
            #task_labels += list(batch['label'].squeeze().data.cpu().numpy())
            group_pred += list(out['group_pred'].data.cpu())
            group_gt += list(out['group_gt'].data.cpu())
            embed += list(out['embs'].data.cpu())

        tsne_z_pred = torch.cat(embed, dim=0)
        tsne_g_pred = torch.cat(group_pred,dim=0)
        tsne_g_gt = torch.cat(group_gt,dim=0)
    return tsne_z_pred, tsne_g_pred, tsne_g_gt
