import logging
import torch.nn as nn
import numpy

from allennlp.common import Params
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder as s2s_e

from fds import FDS
from loss import *
from util import soft_labeling, SoftCrossEntropy
from loss_contra import *




def build_model(args, vocab, pretrained_embs, tasks):
    '''
    Build model according to arguments
    '''
    d_word, n_layers_highway = args.d_word, args.n_layers_highway

    # Build embedding layers
    if args.glove:
        word_embs = pretrained_embs
        train_embs = bool(args.train_words)
    else:
        logging.info("\tLearning embeddings from scratch!")
        word_embs = None
        train_embs = True
    word_embedder = Embedding(vocab.get_vocab_size('tokens'), d_word, weight=word_embs, trainable=train_embs,
                              padding_index=vocab.get_token_index('@@PADDING@@'))
    d_inp_phrase = 0

    token_embedder = {"words": word_embedder}
    d_inp_phrase += d_word
    text_field_embedder = BasicTextFieldEmbedder(token_embedder)
    d_hid_phrase = args.d_hid

    # Build encoders
    phrase_layer = s2s_e.by_name('lstm').from_params(Params({'input_size': d_inp_phrase,
                                                             'hidden_size': d_hid_phrase,
                                                             'num_layers': args.n_layers_enc,
                                                             'bidirectional': True}))
    pair_encoder = HeadlessPairEncoder(vocab, text_field_embedder, n_layers_highway,
                                       phrase_layer, dropout=args.dropout)
    d_pair = 2 * d_hid_phrase

    if args.fds:
        _FDS = FDS(feature_dim=d_pair * 4, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                   start_update=args.start_update, start_smooth=args.start_smooth,
                   kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt)

    # Build model and classifiers
    model = MultiTaskModel(args, pair_encoder, _FDS if args.fds else None)
    build_regressor(tasks, model, d_pair)

    if args.cuda >= 0:
        model = model.cuda()

    return model

def build_regressor(tasks, model, d_pair):
    '''
    Build the regressor
    '''
    for task in tasks:
        d_task =  d_pair * 4
        model.build_regressor(task, d_task)
    return

class MultiTaskModel(nn.Module):
    def __init__(self, args, pair_encoder, FDS=None):
        super(MultiTaskModel, self).__init__()
        self.args = args
        self.pair_encoder = pair_encoder

        self.FDS = FDS
        self.start_smooth = args.start_smooth
        self.max_group_index = int(args.groups - 1)
        self.group_range = args.total_groups/args.groups
        self.tsne = args.tsne
        #self.labels = torch.tensor([], dtype=torch.float)


    def build_regressor(self, task, d_inp):
        if self.args.group_wise:
            groups = int(self.args.groups)
            self.groups = groups
            #for i in range(groups):
            #    layer_ = nn.Linear(d_inp, 1)
            #    setattr(self, 'regressor_%s_pred_layer' % i, layer_)
            setattr(self, 'regressor' , nn.Linear(d_inp, groups) )
            setattr(self, 'classifier' , nn.Linear(d_inp, groups) )
            if self.args.la:
                # TO DO: class_num_list
                cls_num_list = self.get_cls_num_list(self.args)
                #print("+++++++++++++++")
                #print(len(cls_num_list))
                self.lce = LAloss(cls_num_list, tau=self.args.tau).cuda()
            else:
                self.lce = nn.CrossEntropyLoss()
            self.group_range = int(self.args.total_groups/self.args.groups)
        else:
            layer = nn.Linear(d_inp, 1)
            setattr(self, '%s_pred_layer' % task.name, layer)


    def forward(self, task=None, epoch=None, input1=None, input2=None, mask1=None, mask2=None, label=None, weight=None):
        if not self.args.group_wise:
            pred_layer = getattr(self, '%s_pred_layer' % task.name)
        #
        # move tensor to cuda
        #
        input1 = {key : input1[key].cuda() for key in input1}
        input2 = {key : input2[key].cuda() for key in input2}
        label = label.cuda()
        #print(mask1)
        if mask1 is not None and mask2 is not None:
            mask1 = {key: mask1[key].cuda() for key in mask1}
            mask2 = {key : mask2[key].cuda() for key in mask2}
        if weight is not None:
            #print(' after weight ')
            weight = weight.cuda() 
            #weight = {key: weight[key].cuda() for key in weight}
        #
        #print(' before pair encoder ')
        pair_emb = self.pair_encoder(input1, input2, mask1, mask2)
        pair_emb_s = pair_emb

        bsz = pair_emb_s.shape[0]
        out = {}
        if self.args.group_wise:
            #
            # divide groups
            #group_gt = torch.floor(label).to(torch.int)
            group_gt = label/self.group_range
            group_gt = group_gt.to(torch.int)
            group_gt = torch.clamp(group_gt, 0, self.max_group_index)
            #
            cls_layer = getattr(self, 'classifier' )
            group_ = cls_layer(pair_emb_s)
            #
            #print(f' group predition {group_.shape}')
            #
            loss_ce = 0
            if self.args.ce:
                loss_ce = self.lce(group_, group_gt.squeeze(-1).long())
                out['ce'] = loss_ce
            if self.args.ranked_contra:
                #loss_contra = Ranked_Contrastive_Loss(pair_emb, group_gt, self.args.temp) 
                loss_contra =  RnCLoss_groupwise()
                loss_con = loss_contra(pair_emb, group_gt)
                if loss_contra is None:
                    #print('  group gt  ', group_gt)
                    loss_con = 0
                #print(' loss contrastive : ', type(loss_contra), 'group_gt shape is ', group_gt.shape)
                loss_ce += loss_con
                out['loss_contra'] = loss_contra
            if self.args.soft_label:
                group_gt_ = soft_labeling(group_gt.squeeze(-1).long(), self.args).cuda()
                #print('group_ shape : ', group_.shape, 'group_gt shape : ', group_gt.shape)
                loss_ce = SoftCrossEntropy(group_, group_gt_)
                out['ce'] = loss_ce
            # regression
            #pred_list = []
            #pred_list_gt = []
            if self.training:
                #for i in range(bsz):
                #    pred_layer_ = getattr(self, 'regressor' % group_gt[i].item())
                #    pred_list.append(pred_layer_(pair_emb_s[i]))
                reg_pred = self.regressor(pair_emb_s)
                #print(f' reg_pred  is {reg_pred}')
                logits = torch.gather(reg_pred, index=group_gt.to(torch.int64),dim=1)
            else:
                group_hat = torch.argmax(group_, dim=1).unsqueeze(-1)
                reg_pred = self.regressor(pair_emb_s)
                logits = torch.gather(reg_pred, index=group_gt.to(torch.int64),dim=1)
                logits_gt = torch.gather(reg_pred, index=group_hat.to(torch.int64),dim=1)
                #for i in range(bsz):
                #    pred_layer_ = getattr(
                #        self, 'regressor_%s_pred_layer' % group_hat[i].item())
                #    output_ = pred_layer_(pair_emb_s[i])
                #    pred_list.append(output_)
                    # gt
                #    pred_layer_gt = getattr(
                #        self, 'regressor_%s_pred_layer' % group_gt[i].item())
                #    output_gt = pred_layer_gt(pair_emb_s[i])
                #    pred_list_gt.append(output_gt)
                #
                #logits_gt = torch.cat(pred_list_gt)
            #
            #logits = torch.cat(pred_list) 
            #
            #logits = logits.unsqueeze(-1)
            logits = logits.unsqueeze(-1)
            #print(" logits shape ", logits.shape, " label shape ", label.shape )
            #assert logits.shape == label.shape
        else:
            if self.training and self.FDS is not None:
                if epoch >= self.start_smooth:
                    pair_emb_s = self.FDS.smooth(pair_emb_s, label, epoch)
            logits = pred_layer(pair_emb_s)
        ###
        if self.training and self.FDS is not None:
            out['embs'] = pair_emb
            out['labels'] = label
        if self.args.loss == 'huber':
            loss = globals()[f"weighted_{self.args.loss}_loss"](
                inputs=logits, targets=label / torch.tensor(5.).cuda(), weights=weight,
                beta=self.args.huber_beta
            )
        else:
            loss = globals()[f"weighted_{self.args.loss}_loss"](
                inputs=logits, targets=label / torch.tensor(5.).cuda(), weights=weight
            )
            if not self.training and self.args.group_wise :
                loss_gt = globals()[f"weighted_{self.args.loss}_loss"](
                    inputs=logits_gt, targets=label / torch.tensor(5.).cuda(), weights=weight
                )
                out['loss_gt'] = loss_gt
                print(' task cls loss is ', loss_ce.item(),' task reg loss is ', loss.item(), ' gt reg loss is ', loss_gt.item())
                #out['logits_gt'] = logits_gt
        out['logits'] = logits
        label = label.squeeze(-1).data.cpu().numpy()
        logits = logits.squeeze(-1).data.cpu().numpy()
        task.scorer(logits, label)
        if self.args.group_wise:
            out['loss'] = self.args.sigma*loss + loss_ce
            #if not self.args.g_dis:
            #    out['loss'] = self.args.sigma*loss + loss_ce
            #else:
            #    current_loss = self.args.sigma*loss + loss_ce
            #    if current_loss < 1:
            #        out['loss'] = loss + loss_ce
            #    else:
            #        out['loss'] = current_loss
            if not self.training:
                logits_gt = logits_gt.squeeze(-1).data.cpu().numpy()
                task.scorer_gt(logits_gt, label)
        else:
            out['loss'] = loss
        #
        if self.tsne :
            if out.get('embs', 0) != 0:
                out['embs'] = pair_emb
            out['group_pred'] = group_hat
            out['group_gt'] = group_gt 
            
        return out
    
    def get_cls_num_list(self, args, labels = None):
        if labels is None:
            labels = torch.load('labels.pt')
    
        #labels = labels.numpy()
        groups = labels/self.group_range
        groups = groups.to(torch.int)
        groups = torch.clamp(groups, 0, args.groups-1)
        #
        cls_num_dict = {}
        for i in groups :
            key = i.item()
            cls_num_dict[key] = cls_num_dict.get(key, 0)
            cls_num_dict[key] += 1
        #print(" key is ", cls_num_dict.keys())
        #print(" key len is ", len(cls_num_dict.keys()))

        cls_num_list = [cls_num_dict[key] for key in sorted(cls_num_dict.keys())]
        #
        return cls_num_list
    



class HeadlessPairEncoder(Model):
    def __init__(self, vocab, text_field_embedder, num_highway_layers, phrase_layer,
                 dropout=0.2, mask_lstms=True, initializer=InitializerApplicator()):
        super(HeadlessPairEncoder, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        d_emb = text_field_embedder.get_output_dim()
        self._highway_layer = TimeDistributed(Highway(d_emb, num_highway_layers))

        self._phrase_layer = phrase_layer
        self.pad_idx = vocab.get_token_index(vocab._padding_token)
        self.output_dim = phrase_layer.get_output_dim()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self, s1, s2, m1=None, m2=None):
        #
        s1_embs = self._highway_layer(self._text_field_embedder(s1) if m1 is None else s1)
        s2_embs = self._highway_layer(self._text_field_embedder(s2) if m2 is None else s2)

        s1_embs = self._dropout(s1_embs)
        s2_embs = self._dropout(s2_embs)

        # Set up masks
        s1_mask = util.get_text_field_mask(s1) if m1 is None else m1.long()
        s2_mask = util.get_text_field_mask(s2) if m2 is None else m2.long()

        s1_lstm_mask = s1_mask.float() if self._mask_lstms else None
        s2_lstm_mask = s2_mask.float() if self._mask_lstms else None

        # Sentence encodings with LSTMs
        s1_enc = self._phrase_layer(s1_embs, s1_lstm_mask)
        s2_enc = self._phrase_layer(s2_embs, s2_lstm_mask)

        s1_enc = self._dropout(s1_enc)
        s2_enc = self._dropout(s2_enc)

        # Max pooling
        s1_mask = s1_mask.unsqueeze(dim=-1)
        s2_mask = s2_mask.unsqueeze(dim=-1)
        s1_enc.data.masked_fill_(1 - s1_mask.byte().data, -float('inf'))
        s2_enc.data.masked_fill_(1 - s2_mask.byte().data, -float('inf'))
        s1_enc, _ = s1_enc.max(dim=1)
        s2_enc, _ = s2_enc.max(dim=1)

        return torch.cat([s1_enc, s2_enc, torch.abs(s1_enc - s2_enc), s1_enc * s2_enc], 1)
