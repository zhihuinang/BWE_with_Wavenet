import os
from glob import glob

files = [f for f in glob('D:/Master/语音信号处理/Homework/BWE/BWE_dataset/train/clean_trainset_56spk_wav8k/*')]

with open('train.csv','w') as t:
    for f in files:
        f = f.split('/')[-1].split('\\')
        t.write(f[-1])
        t.write('\n')

files = [f for f in glob('D:/Master/语音信号处理/Homework/BWE/BWE_dataset/test/clean_testset_wav8k/*.wav')]

with open('test.csv','w') as t:
    for f in files:
        f = f.split('/')[-1].split('\\')
        t.write(f[-1])
        t.write('\n')