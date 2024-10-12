import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(BasicBlock, self).__init__()
        if base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=3, dropout=None, width_per_group=64):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print(f'Using dropout: {dropout}')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet50': [resnet50, 2048]
}


class Encoder(nn.Module):
    def __init__(self, name='resnet18'):
        super(Encoder, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()

    def forward(self, x):
        feat = self.encoder(x)
        return feat


class SupResNet(nn.Module):
    def __init__(self, name='resnet18', num_classes=10):
        super(SupResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        output = self.fc(feat)

        return output
    

class Encoder_regression(nn.Module):
    def __init__(self, groups=10, name='resnet50', norm=False):
        super(Encoder_regression, self).__init__()
        backbone, dim_in = model_dict[name]
        self.output_dim = groups * 2
        self.encoder = backbone()
        #self.regressor = nn.Sequential(nn.Linear(dim_in, 2048),
        #                               nn.ReLU(),
        #                               nn.Linear(2048, 512),
        #                               nn.ReLU(),
        #                               nn.Linear(512, self.output_dim))
        self.regressor = nn.Sequential(nn.Linear(dim_in, self.output_dim))
        self.norm = norm

    def forward(self, x):
        feat = self.encoder(x)
        if self.norm:
            feat = F.normalize(feat, dim=-1)
        pred = self.regressor(feat)
        return pred, feat
    



class Encoder_regression_single(nn.Module):
    def __init__(self, name='resnet50', norm=False, weight_norm= False):
        super(Encoder_regression_single, self).__init__()
        backbone, dim_in = model_dict[name]
        self.encoder = backbone()
        self.norm = norm
        self.weight_norm = weight_norm
        if self.weight_norm:
            self.regressor = torch.nn.utils.weight_norm(nn.Linear(dim_in, 1), name='weight')
        else:
            self.regressor = nn.Sequential(nn.Linear(dim_in, 1))

    def forward(self, x):
        feat = self.encoder(x)
        if self.norm:
            feat = F.normalize(feat, dim=-1)
        pred = self.regressor(feat)
        return pred, feat
    

# three expert 1) maj 2) med  and 3) low
class Encoder_regression_multi_expert(nn.Module):
    def __init__(self, name='resnet50', norm=False, weight_norm= False):
        super(Encoder_regression_multi_expert, self).__init__()
        backbone, dim_in = model_dict[name]
        self.encoder = backbone()
        self.norm = norm
        self.weight_norm = weight_norm
        if self.weight_norm:
            self.regressor_maj = torch.nn.utils.weight_norm(nn.Linear(dim_in, 1), name='weight')
            self.regressor_med = torch.nn.utils.weight_norm(nn.Linear(dim_in, 1), name='weight')
            self.regressor_min = torch.nn.utils.weight_norm(nn.Linear(dim_in, 1), name='weight')
        else:
            self.regressor_maj = nn.Sequential(nn.Linear(dim_in, 1))
            self.regressor_med = nn.Sequential(nn.Linear(dim_in, 1))
            self.regressor_min = nn.Sequential(nn.Linear(dim_in, 1))
        

    def forward(self, x):
        feat = self.encoder(x)
        if self.norm:
            feat = F.normalize(feat, dim=-1)
        pred_maj = self.regressor_maj(feat)
        pred_med = self.regressor_med(feat)
        pred_min = self.regressor_min(feat)
        return torch.cat((pred_maj, pred_med, pred_min), dim=-1)
    


class Encoder_regression_guided_multi_regression(nn.Module):
    def __init__(self, name='resnet50', norm=False, weight_norm= False):
        super(Encoder_regression_guided_multi_regression, self).__init__()
        backbone, dim_in = model_dict[name]
        self.encoder = backbone()
        self.norm = norm
        self.weight_norm = weight_norm
        if self.weight_norm:
            self.cls_head = torch.nn.utils.weight_norm(nn.Linear(dim_in, 3), name='weight')
            self.regressor_maj = torch.nn.utils.weight_norm(nn.Linear(dim_in, 1), name='weight')
            self.regressor_med = torch.nn.utils.weight_norm(nn.Linear(dim_in, 1), name='weight')
            self.regressor_min = torch.nn.utils.weight_norm(nn.Linear(dim_in, 1), name='weight')
        else:
            self.cls_head = nn.Sequential(nn.Linear(dim_in, 3))
            self.regressor_maj = nn.Sequential(nn.Linear(dim_in, 1))
            self.regressor_med = nn.Sequential(nn.Linear(dim_in, 1))
            self.regressor_min = nn.Sequential(nn.Linear(dim_in, 1))
        

    def forward(self, x):
        feat = self.encoder(x)
        if self.norm:
            feat = F.normalize(feat, dim=-1)
        cls_pred = self.cls_head(feat)
        pred_maj = self.regressor_maj(feat)
        pred_med = self.regressor_med(feat)
        pred_min = self.regressor_min(feat)
        return cls_pred, torch.cat((pred_maj, pred_med, pred_min), dim=-1)
    




class Encoder_regression_uncertainty(nn.Module):
    def __init__(self, name='resnet50', norm=False, weight_norm= False):
        super(Encoder_regression_uncertainty, self).__init__()
        backbone, dim_in = model_dict[name]
        self.encoder = backbone()
        self.norm = norm
        self.weight_norm = weight_norm
        if self.weight_norm:
            self.regressor = torch.nn.utils.weight_norm(nn.Linear(dim_in, 2), name='weight')
        else:
            self.regressor = nn.Linear(dim_in, 2)
        

    def forward(self, x):
        feat = self.encoder(x)
        if self.norm:
            feat = F.normalize(feat, dim=-1)
        out = self.regressor(feat)
        out_ = torch.chunk(out,2,dim=1)
        return out_[0], out_[1]


 
    
class Regression_guassian_likelihood(nn.Module):
    def __init__(self, name='resnet50', norm=False, weight_norm= False):
        super(Regression_guassian_likelihood, self).__init__()
        backbone, dim_in = model_dict[name]
        self.encoder = backbone()
        self.norm = norm
        self.weight_norm = weight_norm
        #if self.weight_norm:
        #    self.regressor = torch.nn.utils.weight_norm(nn.Linear(dim_in, 2), name='weight')
        #else:
        #   self.regressor = nn.Linear(dim_in, 2)
        self.guassuann_head = GaussianLikelihoodHead(inp_dim=dim_in, outp_dim=1)
        

    def forward(self, x):
        feat = self.encoder(x)
        if self.norm:
            feat = F.normalize(feat, dim=-1)
        out = self.guassuann_head(feat)
        out_ = torch.chunk(out,2,dim=1)
        return out_[0], out_[1]


class GaussianLikelihoodHead(nn.Module):
    def __init__(
        self,
        inp_dim,
        outp_dim,
        initial_var=1,
        min_var=1e-8,
        max_var=100,
        mean_scale=1,
        var_scale=1,
        use_spectral_norm_mean=False,
        use_spectral_norm_var=False,
    ):
        super().__init__()
        assert min_var <= initial_var <= max_var

        self.min_var = min_var
        self.max_var = max_var
        self.init_var_offset = np.log(np.exp(initial_var - min_var) - 1)

        self.mean_scale = mean_scale
        self.var_scale = var_scale

        if use_spectral_norm_mean:
            self.mean = nn.utils.spectral_norm(nn.Linear(inp_dim, outp_dim))
        else:
            self.mean = nn.Linear(inp_dim, outp_dim)

        if use_spectral_norm_var:
            self.var = nn.utils.spectral_norm(nn.Linear(inp_dim, outp_dim))
        else:
            self.var = nn.Linear(inp_dim, outp_dim)

    def forward(self, inp):
        mean = self.mean(inp) * self.mean_scale
        var = self.var(inp) * self.var_scale

        var = F.softplus(var + self.init_var_offset) + self.min_var
        var = torch.clamp(var, self.min_var, self.max_var)

        return mean, var
