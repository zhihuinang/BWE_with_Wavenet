import os
import torch
import random
import yaml
import numpy as np
import time
import argparse
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import Adam

from model.X_net import *
from model.dilation_X_net import *
from dataset.dataloader import *
from utils import *

logger = logging.getLogger(__name__)




def eval(model,test_loader):
    global_step = 0
    t_total = len(test_loader)
    
    total_loss = 0
    total_lsd = 0
    total_snr = 0
    total_pesq = 0
    model.zero_grad()
    model.eval()
    epoch_iterator = tqdm(test_loader,
                            desc="Training (X / X Steps) (loss=X.X)",
                            bar_format="{l_bar}{r_bar}",
                            dynamic_ncols=True)
    for data in epoch_iterator:
        Hi_audio = data['Hi_audio']
        Lo_audio = data['Lo_audio']
        Hi_res = model(Lo_audio,mode='up')
        loss = T_MSE_Loss(Hi_res,Hi_audio)
    
        epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, loss.item()))

        total_loss += loss.item()
        global_step += 1
        
        lsd = LSD(Hi_res,Hi_audio)
        snr = SNR(Hi_res,Hi_audio)
        pseq = PESQ(Hi_res,Hi_audio)
        total_lsd += lsd
        total_snr += snr
        total_pesq += pseq

    Lsd = total_lsd / len(test_loader)
    Snr = total_snr / len(test_loader)
    Pesq = total_pesq / len(test_loader)
    
    logger.info('finish evaluation, LSD = {}, SNR = {}, PESQ = {}'.format(Lsd, Snr, Pesq))
    total_lsd = 0
   
    logger.info("--------------------Finish evaluation--------------------")

    return 
    


def get_dataloader(file,mode='train'):
    sample_list = []
    with open(file,'r') as f:
        for line in f:
            sample_list.append(line.strip(''))

    bwe_dataset = BWE_dataset(sample_list,mode=mode)
    if mode == 'train':    
        data_loader = DataLoader(bwe_dataset,
                            batch_size = 24,
                            shuffle=True,
                            drop_last=True,
                            num_workers = 0)
    else: 
        data_loader = DataLoader(bwe_dataset,
                            batch_size = 24,
                            shuffle=False,
                            drop_last=True,
                            num_workers = 0)
    return data_loader



def main(config):

    ckpt = config['ckpt']
    
    test_file = config['test_file']

    
    test_loader = get_dataloader(test_file,mode='test')

    model_type = config['model_type']

    if model_type == 'X-net':
        model = X_net()
      
    elif model_type == 'Dilation_X-net':
        model = Dilation_X_net()
    
    if os.path.isfile('{}/model_ph1.pth'.format(ckpt)):
        logger.info("------resuming last training------")
        checkpoint = torch.load('{}/model_ph2.pth'.format(ckpt),map_location='cpu')
        model.load_state_dict(checkpoint['net'])
    

    
    eval(model,test_loader)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    
    
    config_path = args.config

    if os.path.isfile(config_path):
        f = open(config_path)
        config = yaml.load(f,Loader=yaml.FullLoader)
        print("***********************************")
        print(yaml.dump(config, default_flow_style=False, default_style=''))
        print("***********************************")
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

    random.seed(1234)
    torch.manual_seed(1234)

    ckpt = config['ckpt']
    os.makedirs(ckpt,exist_ok=True)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s',datefmt='%m/%d/%Y %H:%M:%S')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(ckpt+'/result.log',encoding='utf8',mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    main(config)


