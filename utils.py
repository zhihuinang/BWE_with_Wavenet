import math
import torch
import librosa
import logging
import os
import numpy as np
import time
import soundfile as sf
from pesq import pesq

logger = logging.getLogger(__name__)

def inv_mulaw(y, mu=256):
    r"""Inverse of mu-law companding (mu-law expansion)
    Args:
        y (array-like): Compressed signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Uncomprresed signal (-1 <= x <= 1)
    """
    return torch.sign(y) * (1.0 / mu) * ((1.0 + mu) ** torch.abs(y) - 1.0)

def mulaw(x, mu=256):
    r"""Mu-Law companding
    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Compressed signal ([-1, 1])
    """
    return torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(torch.tensor(mu))

    

def mulaw_quantize(x, mu=256):
    """Mu-Law companding + quantize
    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Quantized signal (dtype=int)
          - y ∈ [0, mu] if x ∈ [-1, 1]
          - y ∈ [0, mu) if x ∈ [-1, 1)
    .. note::
        If you want to get quantized values of range [0, mu) (not [0, mu]),
        then you need to provide input signal of range [-1, 1).
    """
    y = mulaw(x, mu)
    # scale [-1, 1] to [0, mu]
    return torch.floor((y + 1) / 2 * mu)

def inv_mulaw_quantize(y, mu=256):
    r"""Inverse of mu-law companding + quantize
    Args:
        y (array-like): Quantized signal (∈ [0, mu]).
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Uncompressed signal ([-1, 1])
    """
    # [0, m) to [-1, 1]
    y = 2 * y.float() / mu - 1
    return inv_mulaw(y, mu)


def LSD(output,target):
    output = output.detach()
    target = target.detach()
    spec_output = torch.log10(torch.abs(torch.stft(output,2048,return_complex=True)).clamp(1e-10,999999))
    spec_target = torch.log10(torch.abs(torch.stft(target,2048,return_complex=True)).clamp(1e-10,999999))
    distants =torch.sqrt(torch.mean((spec_output-spec_target)**2,dim=1)) 
    B,L = distants.shape
    lsd = torch.sum(distants,dim=1) * 2 / L
    return torch.mean(lsd)



def SNR(output,target):
    output = output.detach()
    target = target.detach()
    
    dif = (target-output).norm(p=2,dim=1)
    y_2 = target.norm(p=2,dim=1)
    
    snr = 10*torch.log2(y_2/dif)
    return torch.mean(snr)


def PESQ(output,target):
    output = output.detach().numpy()
    target = target.detach().numpy()
    p_total = 0
    B = output.shape[0]
    for i in range(B):
        o = output[i,:]
        t = target[i,:]
        try:
            p = pesq(16000,t,o,'wb')
        except Exception:
            p=0
        p_total += p
    return p_total/B

    

def save_checkpoint(model, config,name):
    checkpoint = {
            "net": model.state_dict(keep_vars=True),
        }
    ckpt = config["ckpt"]
    os.makedirs(ckpt,exist_ok=True)
    torch.save(checkpoint,
                  '{}/model_{}.pth'.format(ckpt,name))
    

def T_MSE_Loss(output,target):
    return torch.nn.functional.mse_loss(output,target)

def F_MSE_Loss(output,target):
    spec_output = torch.abs(torch.stft(output,2048,return_complex=True))
    spec_target = torch.abs(torch.stft(target,2048,return_complex=True))

    return torch.nn.functional.mse_loss(spec_output,spec_target)
    

def write_audio(Hi_res,fname,config):
    Hi_res = Hi_res.detach().numpy()
    ckpt = config['ckpt']+'/demo/'
    os.makedirs(ckpt,exist_ok=True)
    sf.write(ckpt+fname[0],Hi_res,samplerate=16000)