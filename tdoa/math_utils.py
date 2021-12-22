import numpy as np
import torch
import torch.fft

SPEED_OF_SOUND = 343.0


def compute_distance(p1, p2, mode="numpy"):
    "Compute the euclidean distance between two points"

    if mode == "numpy":
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.linalg.norm(p1 - p2)
    elif mode == "torch":
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


def denormalize(x, min_x, max_x):
    return x*(max_x - min_x) + min_x


def gcc_phat(x1, x2, fs):
    """
    This function computes the offset between the signal sig and the reference signal x2
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT) method.
    Implementation based on http://www.xavieranguera.com/phdthesis/node92.html
    """
    
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


def compute_doa(m1, m2, s, radians=True):
    """Get the direction of arrival between two microphones and a source.
       The referential used is the direction of the two sources, that is,
       the vector m1 - m2.

       For more details, see: 
       https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors/879474

    Args:
        m1 (np.array): 2d coordinates of microphone 1
        m2 (np.array): 2d coordinates of microphone 2
        s (np.array): 2d coordinates of the source
        radians (bool): If True, result is between [-pi, pi). Else, result is between [0, 360)
    """

    reference_direction = m1 - m2
    mic_centre = (m1 + m2)/2
    source_direction = s - mic_centre

    doa = compute_angle_between_vectors(reference_direction, source_direction,
                                        radians=radians)

    return doa


def compute_angle_between_vectors(v1, v2, radians=True):
    dot = np.dot(v1, v2)
    det = np.linalg.det([v1, v2])

    doa = np.arctan2(det, dot)

    if not radians:
        doa = np.rad2deg(doa)
    
    return doa
