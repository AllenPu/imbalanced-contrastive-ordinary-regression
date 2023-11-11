import torch.nn as nn
import torchvision
import torch
import torch.optim as optim


class ResNet_regression(nn.Module):
    def __init__(self, args=None):
        super(ResNet_regression, self).__init__()
        self.args = args
        self.groups = args.groups
        exec('self.model = torchvision.models.resnet{}(pretrained=False)'.format(args.model_depth))
        #
        output_dim = args.groups * 2
        #
        fc_inputs = self.model.fc.in_features
        #
        self.model_extractor = nn.Sequential(*list(self.model.children())[:-1])
        #
        self.Flatten = nn.Flatten(start_dim=1)
        #
        self.model_linear =  nn.Sequential(nn.Linear(fc_inputs, output_dim))
        #

        #self.mode = args.mode
        self.sigma = args.sigma
        
    # g is the same shape of y
    def forward(self, x):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #
        z = self.Flatten(z)
        #
        y_hat = self.model_linear(z)
        #
        # the ouput dim of the embed is : 512
        #
        return y_hat, z
    

def opts(self, model, args, split_opt=False):
    opt_list = []
    if split_opt:
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        opt_all = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
