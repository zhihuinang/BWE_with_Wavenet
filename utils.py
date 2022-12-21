import math
import torch
import librosa
import logging
import os
import numpy as np


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
    output = np.array(output)
    target = np.array(target)
    target_spectrogram = librosa.core.stft(target, n_fft=2048)
    output_spectrogram = librosa.core.stft(output, n_fft=2048)
 
    target_log = np.log10(np.abs(target_spectrogram) ** 2)
    output_log = np.log10(np.abs(output_spectrogram) ** 2)
    output_target_squared = (output_log - target_log) ** 2
    lsd = np.mean(np.sqrt(np.mean(output_target_squared, axis=1)))
 
    return lsd






def save_checkpoint(model, config):
    checkpoint = {
            "net": model.state_dict(keep_vars=True),
        }
    ckpt = config["ckpt"]
    os.makedirs(ckpt,exist_ok=True)
    torch.save(checkpoint,
                  '{}/model.pth'.format(ckpt))
    logger.info("Saved model checkpoint to [DIR: %s]", ckpt)

