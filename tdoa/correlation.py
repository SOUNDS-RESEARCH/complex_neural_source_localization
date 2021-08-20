import numpy as np
import numpy.fft as fft
from scipy.signal.signaltools import correlate, correlation_lags


def compute_correlations(simulation_results, fs, mode="gcc-phat"):
    n_microphones = simulation_results.shape[0]

    correlations = {}
    for n_microphone_1 in range(n_microphones):
        for n_microphone_2 in range(n_microphone_1 + 1, n_microphones):
            key = (n_microphone_1, n_microphone_2)
            cc, lag_indexes = _compute_correlation(
                simulation_results[n_microphone_1],
                simulation_results[n_microphone_2],
                fs,
                mode=mode
            )
            tdoa = lag_indexes[np.argmax(np.abs(cc))]
            correlations[key] = tdoa, cc

    return correlations



def _compute_correlation(x1, x2, fs, mode="gcc-phat"):
    if mode == "gcc-phat":
        cc, lag_indexes = gcc_phat(x1, x2, fs)
    else:
        cc, lag_indexes = temporal_cross_correlation(x1, x2, fs)

    return cc, lag_indexes


def temporal_cross_correlation(x1, x2, fs):
    
    # Normalize signals for a normalized correlation
    # https://github.com/numpy/numpy/issues/2310
    x1 = (x1 - np.mean(x1)) / (np.std(x1) * len(x1))
    x2 = (x2 - np.mean(x2)) / (np.std(x2) * len(x2))
    
    cc = correlate(x1, x2, mode="same")
    lag_indexes = correlation_lags(x1.shape[0], x2.shape[0], mode="same")

    cc = np.abs(cc)
    
    return cc, lag_indexes/fs


def gcc_phat(x1, x2, fs):
    '''
    This function computes the offset between the signal sig and the reference signal x2
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT) method.
    Implementation based on http://www.xavieranguera.com/phdthesis/node92.html
    '''
    
    n = x1.shape[0] # + x2.shape[0]

    X1 = np.fft.rfft(x1, n=n)
    X2 = np.fft.rfft(x2, n=n)
    R = X1 * np.conj(X2)
    Gphat = R / np.abs(R)
    cc = np.fft.irfft(Gphat, n=n)

    max_shift = n // 2

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    
    indxs = np.zeros_like(cc)
    indxs[0:max_shift] = - np.arange(max_shift, 0, -1)
    indxs[max_shift:] = np.arange(0, max_shift + 1)
    indxs = indxs/fs

    return cc, indxs