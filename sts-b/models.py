import logging
import torch.nn as nn

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

        self.args = args

    def build_regressor(self, task, d_inp):
        if self.args.group_wise:
            layer = nn.Linear(d_inp, 1)
            groups = int(self.args.groups)
            self.groups = groups
            for i in range(groups):
                setattr(self, 'regressor_%s_pred_layer' % i, layer)
            setattr(self, 'classifier' , nn.Linear(d_inp, groups) )
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
        if mask1 and mask2 :
            mask1 = {key: mask1[key].cuda() for key in mask1}
            mask2 = {key : mask2[key].cuda() for key in mask2}
        if weight:
            weight = weight.cuda() 
            #weight = {key: weight[key].cuda() for key in weight}
        #
        pair_emb = self.pair_encoder(input1, input2, mask1, mask2)
        pair_emb_s = pair_emb

        bsz = pair_emb_s.shape[0]

        if self.args.group_wise:
            logit_out = []
            group_gt = (label/self.group_range).to(torch.int)
            #
            cls_layer = getattr(self, 'classifier' )
            group_ = cls_layer(pair_emb_s)
            loss_ce = self.lce(group_, group_gt.squeeze(-1).long())
            # regression
            pred_list = []
            pred_list_gt = []
            if self.training :
                for i in range(bsz):
                    pred_layer_ = getattr(self, 'regressor_%s_pred_layer' % group_gt[i].item())
                    pred_list.append(pred_layer_(pair_emb_s[i]))
            else:
                group_hat = torch.argmax(group_, dim=1).unsqueeze(-1)
                for i in range(bsz):
                    pred_layer_ = getattr(
                        self, 'regressor_%s_pred_layer' % group_hat[i].item())
                    pred_list.append(pred_layer_(pair_emb_s[i]))
                    # gt
                    pred_layer_gt = getattr(
                        self, 'regressor_%s_pred_layer' % group_gt[i].item())
                    pred_list_gt.append(pred_layer_gt(pair_emb_s[i]))
                logits_gt = torch.cat(pred_list_gt)

            #
            logits = torch.cat(pred_list) 
            #
            logits = logits.unsqueeze(-1)
            print(" logits shape ", logits.shape, " label shape ", label.shape )
            assert logits.shape == label.shape
        else:
            if self.training and self.FDS is not None:
                if epoch >= self.start_smooth:
                    pair_emb_s = self.FDS.smooth(pair_emb_s, label, epoch)
            logits = pred_layer(pair_emb_s)


        out = {}
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
            if not self.training:
                loss_gt = globals()[f"weighted_{self.args.loss}_loss"](
                    inputs=logits_gt, targets=label / torch.tensor(5.).cuda(), weights=weight
                )
                out['loss_gt'] = loss_gt
                #out['logits_gt'] = logits_gt
        out['logits'] = logits
        label = label.squeeze(-1).data.cpu().numpy()
        logits = logits.squeeze(-1).data.cpu().numpy()
        task.scorer(logits, label)
        if self.args.group_wise:
            if not self.args.g_dis:
                out['loss'] = self.args.sigma*loss + loss_ce
            else:
                current_loss = self.args.sigma*loss + loss_ce
                if current_loss < 1:
                    out['loss'] = loss + loss_ce
                else:
                    out['loss'] = current_loss

            if not self.training:
                logits_gt = logits_gt.squeeze(-1).data.cpu().numpy()
                task.scorer_gt(logits_gt, label)
        else:
            out['loss'] = loss

        return out
    



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
