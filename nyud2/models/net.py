import torch
import torch.nn as nn
from models import modules


class model(nn.Module):
    def __init__(self, args, Encoder, num_features, block_channel):

        super(model, self).__init__()

        self.E = Encoder
        self.D = modules.D(num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R(args, block_channel)
        #self.classifier = modules.classifier_Regressor(args)
        self.input_shape = 0
        self.args = args

    def forward(self, x, depth=None, epoch=None):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4, [
                         x_decoder.size(2), x_decoder.size(3)])
        if self.args.group_mode and self.input_shape == 0:
            feat = torch.cat((x_decoder, x_mff), 1)
            reg_output = self.R(feat, depth, epoch)
            cls_output = self.classifier(torch.cat((x_decoder, x_mff), 1))
            if self.input_shape == 0:
                self.input_shape = self.R.input_shape
                self.classifier = modules.classifier_Regressor(self.args, self.input_shape)
                cls_output = self.classifier(feat)
            else:
                cls_output = self.classifier(feat)
            return reg_output, cls_output
        else:
            out = self.R(torch.cat((x_decoder, x_mff), 1), depth, epoch)
            return out
