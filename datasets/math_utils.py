import numpy as np
import torch

SPEED_OF_SOUND = 343.0

def compute_distance(p1, p2, mode="numpy"):
    "Compute the euclidean distance between two points"

    if mode == "numpy":
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.linalg.norm(p1 - p2)
    elif mode == "torch":
        p1 = torch.Tensor(p1)
        p2 = torch.Tensor(p2)
        return torch.linalg.norm(p1 - p2)


def compute_tdoa(source, microphones):
    dist_0 = compute_distance(source, microphones[0])
    dist_1 = compute_distance(source, microphones[1])

    return (dist_0 - dist_1)/SPEED_OF_SOUND


def normalize_tdoa(tdoa, mic_distance):
    max_tdoa = mic_distance/SPEED_OF_SOUND
    min_tdoa = -max_tdoa

    return normalize(tdoa, min_tdoa, max_tdoa)


def normalize(x, min_x, max_x):
    return (x - min_x)/(max_x - min_x)


def gcc_phat(x1, x2, fs):
    '''
    This function computes the offset between the signal sig and the reference signal x2
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT) method.
    Implementation based on http://www.xavieranguera.com/phdthesis/node92.html
    '''
    
    n = x1.shape[0] # + x2.shape[0]

    X1 = torch.fft.rfft(x1, n=n)
    X2 = torch.fft.rfft(x2, n=n)
    R = X1 * torch.conj(X2)
    Gphat = R / torch.abs(R)
    cc = torch.fft.irfft(Gphat, n=n)

    max_shift = n // 2

    cc = torch.cat((cc[-max_shift:], cc[:max_shift+1]))
    
    indxs = torch.zeros_like(cc)
    indxs[0:max_shift] = - torch.arange(max_shift, 0, -1)
    indxs[max_shift:] = torch.arange(0, max_shift + 1)
    indxs = indxs/fs

    return cc, indxs