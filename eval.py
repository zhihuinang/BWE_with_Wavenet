import os
import torch
import random
import yaml
import numpy as np
import argparse
import logging
from tqdm import tqdm
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam

from model.wavenet import *
from dataset.dataloader import *
from utils import *

logger = logging.getLogger(__name__)


def evaluate(model,test_loader,config):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    t_total = len(test_loader)
    total_loss = 0
    epoch_iterator = tqdm(test_loader,
                              desc="Testing (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
    for data in epoch_iterator:
        input_data = data['data']
        target = data['target'].squeeze()
        output = model(input_data)
        loss = criterion(output,target)
        epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, loss.item()))
        
        total_loss += loss.item()
        global_step += 1
    ppl = np.exp(total_loss/len(test_loader))
    logger.info('finish test, ppl = {} for label'.format(ppl))
        


def train(model,optimizer,train_loader,config):
    epoch = config['epoch']
    criterion = nn.CrossEntropyLoss()
    best_Lsd = 9999999
    global_step = 0
    t_total = epoch * len(train_loader)
    

    for i in range(epoch):
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
            Lo_spec = data['Lo_spec']

            Hi_audio = Hi_audio / (2**15)
            input_Hi_audio = mulaw_quantize(Hi_audio)

            output = model(input_Hi_audio,Lo_spec)
            loss = criterion(output,Hi_audio)
            epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, loss.item()))
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            ori_output = inv_mulaw_quantize(ori_output)*(2**15)
            lsd = LSD(ori_output,Hi_audio)
            total_lsd += lsd

        Lsd = total_loss / len(train_loader)
        logger.info('finish training {} epoch, ppl = {} for label'.format(i+1, Lsd))
        total_lsd = 0
        if Lsd < best_Lsd:
            best_Lsd = Lsd
            save_checkpoint(model,config)

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
                            num_workers = 4)
    else: 
        data_loader = DataLoader(bwe_dataset,
                            batch_size = 48,
                            shuffle=False,
                            drop_last=True,
                            num_workers = 4)
    return data_loader



def main(config):

    ckpt = config['ckpt']
    
    test_file = config['test_file']
    
    test_loader = get_dataloader(test_file,mode='test')
    
    model = WaveNet(channels_in=1,channels_out=256)
    
    
    if os.path.isfile('{}/model.pth'.format(ckpt)):
        logger.info("------resuming last training------")
        checkpoint = torch.load('{}/model/model.pth'.format(ckpt),map_location='cpu')
        model.load_state_dict(checkpoint['net'])
    
    
    evaluate(model, test_loader,config)



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











