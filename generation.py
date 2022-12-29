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
import librosa
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import Adam

from model.X_net import *
from model.dilation_X_net import *
from dataset.dataloader import *
from utils import *

logger = logging.getLogger(__name__)




def eval(model,test_loader):
    model.zero_grad()
    model.eval()
    epoch_iterator = tqdm(test_loader,
                            desc="Training (X / X Steps) (loss=X.X)",
                            bar_format="{l_bar}{r_bar}",
                            dynamic_ncols=True)
    for data in epoch_iterator:
        Lo_audio = data['Lo_audio']
        Hi_audio = data['Hi_audio']
        _,Hi_res = model(Hi_audio)
        fname = data['file_name']

        write_audio(Hi_res,fname,config)
    return 
    
class BWE_demo_dataset(Dataset):
    def __init__(self,list_sample,mode='train'):
        super().__init__()
        self.list_sample = list_sample
        self.mode = mode
        

    def __len__(self):
        return len(self.list_sample)

    def __getitem__(self, index):
        file_name = self.list_sample[index].strip()
        if self.mode == 'train':
            Hi_file_name = '../BWE_dataset/train/clean_trainset_56spk_wav16k/'+file_name
            Lo_file_name = '../BWE_dataset/train/clean_trainset_56spk_wav8k/'+file_name

        else:
            Hi_file_name = '../BWE_dataset/test/clean_testset_wav16k/'+file_name
            Lo_file_name = '../BWE_dataset/test/clean_testset_wav8k/'+file_name

        Hi_audio, sr_Hi = librosa.load(Hi_file_name,sr=None)
        Lo_audio, sr_Lo = librosa.load(Lo_file_name,sr=None)

        
        return {'Hi_audio':Hi_audio,'Lo_audio':Lo_audio,'file_name':file_name}


def get_dataloader(file,mode='train'):
    sample_list = []
    with open(file,'r') as f:
        for line in f:
            sample_list.append(line.strip(''))

    bwe_dataset = BWE_demo_dataset(sample_list,mode=mode)
    if mode == 'train':    
        data_loader = DataLoader(bwe_dataset,
                            batch_size = 24,
                            shuffle=True,
                            drop_last=True,
                            num_workers = 0)
    else: 
        data_loader = DataLoader(bwe_dataset,
                            batch_size = 1,
                            shuffle=False,
                            drop_last=True,
                            num_workers = 0)
    return data_loader



def main(config):

    ckpt = config['ckpt']
    
    test_file = './data/test_demo.csv'

    
    test_loader = get_dataloader(test_file,mode='test')

    model_type = config['model_type']

    if model_type == 'X-net':
        model = X_net()
      
    elif model_type == 'Dilation_X-net':
        model = Dilation_X_net()
    
    if os.path.isfile('{}/model_ph1.pth'.format(ckpt)):
        logger.info("------resuming last training------")
        checkpoint = torch.load('{}/model_ph1.pth'.format(ckpt),map_location='cpu')
        model.load_state_dict(checkpoint['net'])
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



    main(config)


