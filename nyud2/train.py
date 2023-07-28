import argparse
import time
import os
import shutil
import logging
import torch
import torch.backends.cudnn as cudnn
import nyud2.loaddata
from tqdm import tqdm
import nyud2.models.modules as modules
import nyud2.models.net as net
import nyud2.models.resnet as resnet
from nyud2.util import query_yes_no
from test import test
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='')

def define_model(args):
    original_model = resnet.resnet50(pretrained=True)
    Encoder = modules.E_resnet(original_model)
    model = net.model(args, Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model



