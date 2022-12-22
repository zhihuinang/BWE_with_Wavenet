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
from dataset.dataloader import *
from utils import *

logger = logging.getLogger(__name__)




def train(model,optimizer,train_loader,config):
    epoch = config['epoch']
    best_Lsd = 9999999
    global_step = 0
    t_total = epoch[0] * len(train_loader)
    

    for i in range(epoch[0]):
        total_loss = 0
        total_lsd = 0
        model.zero_grad()
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for data in epoch_iterator:
            Hi_audio = data['Hi_audio']
            Lo_audio = data['Lo_audio']
            Lo_res,Hi_res = model(Hi_audio)
            loss_lo = T_MSE_Loss(Lo_res,Lo_audio)
            loss_hi = T_MSE_Loss(Hi_res,Hi_audio)
            loss = 0.5 * loss_lo + 0.5 * loss_hi
            epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, loss.item()))
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
          
            lsd = LSD(Hi_res,Hi_audio)
            total_lsd += lsd

        Lsd = total_lsd / len(train_loader)
        err = total_loss/ len(train_loader)
        logger.info('finish training {} epoch, Loss = {}, LSD = {}'.format(i+1, err, Lsd))
        total_lsd = 0
        if Lsd < best_Lsd:
            best_Lsd = Lsd
            save_checkpoint(model,config,'ph1')
            logger.info("Saved model checkpoint to [DIR: %s]", config['ckpt'])

    logger.info("--------------------Finish training Phase 1--------------------")
    best_Lsd = 9999999
    global_step = 0
    t_total = epoch[1] * len(train_loader)
    

    for i in range(epoch[1]):
        total_loss = 0
        total_lsd = 0
        model.zero_grad()
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for data in epoch_iterator:
            Hi_audio = data['Hi_audio']
            Lo_audio = data['Lo_audio']
            
            Lo_res,Hi_res = model(Hi_audio)
            loss_lo = F_MSE_Loss(Lo_res,Lo_audio)
            loss_hi = F_MSE_Loss(Hi_res,Hi_audio)
            loss = 0.5 * loss_lo + 0.5 * loss_hi
            epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, loss.item()))
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
          

            lsd = LSD(Hi_res,Hi_audio)
            total_lsd += lsd

        Lsd = total_lsd / len(train_loader)
        err = total_loss/ len(train_loader)
        logger.info('finish training {} epoch, Loss = {}, LSD = {}'.format(i+1, err, Lsd))
        total_lsd = 0
        if Lsd < best_Lsd:
            best_Lsd = Lsd
            save_checkpoint(model,config,'ph2')
            logger.info("Saved model checkpoint to [DIR: %s]", config['ckpt'])
    logger.info("--------------------Finish training Phase 2--------------------")
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
                            batch_size = 8,
                            shuffle=False,
                            drop_last=True,
                            num_workers = 0)
    return data_loader



def main(config):

    ckpt = config['ckpt']
    
    train_file = config['train_file']

    
    train_loader = get_dataloader(train_file,mode='train')

    
    model = X_net()
    
    
    if os.path.isfile('{}/model_ph1.pth'.format(ckpt)):
        logger.info("------resuming last training------")
        checkpoint = torch.load('{}/model_ph1.pth'.format(ckpt),map_location='cpu')
        model.load_state_dict(checkpoint['net'])
        checkpoint = torch.load('{}/model_ph2.pth'.format(ckpt),map_location='cpu')
        model.load_state_dict(checkpoint['net'])
    

    
    optimizer = Adam(model.parameters(),lr = 1e-4)

    
    train(model,optimizer,train_loader,config)




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


