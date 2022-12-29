import torch
import numpy as np
import librosa
import warnings 
from torch.utils.data import Dataset


warnings.filterwarnings('ignore')

class BWE_dataset(Dataset):
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
            sample_time = 350 #ms
        else:
            Hi_file_name = '../BWE_dataset/test/clean_testset_wav16k/'+file_name
            Lo_file_name = '../BWE_dataset/test/clean_testset_wav8k/'+file_name
            sample_time = 2000 #ms

        Hi_audio, sr_Hi = librosa.load(Hi_file_name,sr=None)
        Lo_audio, sr_Lo = librosa.load(Lo_file_name,sr=None)

        
        sample_num_Hi = int(sample_time * sr_Hi / 1000)
        sample_num_Lo = int(sample_time * sr_Lo / 1000)
        if len(Lo_audio) <= sample_num_Lo:
            pad_Lo = sample_num_Lo - len(Lo_audio)
            pad_Hi = sample_num_Hi - len(Hi_audio)
            Lo_audio = np.pad(Lo_audio,(0,pad_Lo),'constant')
            Hi_audio = np.pad(Hi_audio,(0,pad_Hi),'constant')
        else:
            Lo_audio = Lo_audio[:sample_num_Lo]
            Hi_audio = Hi_audio[:sample_num_Hi]
        #hop_length = int(15 * sr_Lo /1000)
        #win_length = int(50 * sr_Lo /1000)
        
        #Lo_spec = librosa.feature.melspectrogram(Lo_audio,sr=sr_Lo,hop_length=hop_length,win_length=win_length,n_mels=64)

        return {'Hi_audio':Hi_audio,'Lo_audio':Lo_audio}
