import librosa
import librosa.display
import matplotlib.pylab as plt
import numpy as np

fname = ['p232_131.wav','p232_148.wav','p232_169.wav']

Lo_dir = '../BWE_dataset/test/clean_testset_wav8k/'
Hi_dir = '../BWE_dataset/test/clean_testset_wav16k/'

X_net_dir = './result/X_net_new/demo/'
Dilation_X_net_dir = './result/Dilation_X_net_new/demo/'

for i, file_name in enumerate(fname):
    Lo_audio,_ = librosa.load(Lo_dir+file_name,sr=None)
    Lo_spec = librosa.stft(Lo_audio)
    librosa.display.specshow(librosa.power_to_db(Lo_spec, ref=np.max))
    plt.savefig('./vis/Lo_{}.png'.format(i), bbox_inches=None, pad_inches=0)
    plt.close()



    Hi_audio,_ = librosa.load(Hi_dir+file_name,sr=None)
    Hi_spec = librosa.stft(Hi_audio)
    librosa.display.specshow(librosa.power_to_db(Hi_spec, ref=np.max))
    plt.savefig('./vis/Hi_{}.png'.format(i), bbox_inches=None, pad_inches=0)
    plt.close()

    X_audio,_ = librosa.load(X_net_dir+file_name,sr=None)
    X_spec = librosa.stft(X_audio)
    librosa.display.specshow(librosa.power_to_db(X_spec, ref=np.max))
    plt.savefig('./vis/X_{}.png'.format(i), bbox_inches=None, pad_inches=0)
    plt.close()
    
    DX_audio,_ = librosa.load(Dilation_X_net_dir+file_name,sr=None)
    DX_spec = librosa.stft(DX_audio)
    librosa.display.specshow(librosa.power_to_db(DX_spec, ref=np.max))
    plt.savefig('./vis/DX_{}.png'.format(i), bbox_inches=None, pad_inches=0)
    plt.close()
