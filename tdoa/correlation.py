import numpy as np
import numpy.fft as fft
from scipy.signal.signaltools import correlate, correlation_lags


def compute_correlations(simulation_results, fs, mode="gcc-phat"):
    n_microphones = simulation_results.shape[0]

    correlations = {}
    for n_microphone_1 in range(n_microphones):
        for n_microphone_2 in range(n_microphone_1 + 1, n_microphones):
            key = (n_microphone_1, n_microphone_2)
            correlations[key] = _compute_correlation(
                simulation_results[n_microphone_1],
                simulation_results[n_microphone_2],
                fs,
                mode=mode
            )

    return correlations



def _compute_correlation(x1, x2, fs, mode="gcc-phat"):
    if mode == "gcc-phat":
        tau, cc = _gcc_phat_new(x1, x2, fs)
    else:
        tau, cc = _cross_correlation(x1, x2, fs)

    return tau, cc


def _cross_correlation(x1, x2, fs):
    cc = correlate(x1, x2, mode="same")
    lag_indexes = correlation_lags(x1.shape[0], x2.shape[0], mode="same")

    tau_samples = lag_indexes[cc.argmax()]
    tau_seconds = tau_samples/fs

    # Return the absolute value: Maybe losing some important information on "which arrived first" here
    return abs(tau_seconds), cc


def _gcc_phat(x1, x2):
    '''
    Retrieve the generalized Cross Correlation - Phase Transform (GCC-PHAT)
    for a pair of microphones. 
    
    Reference: 
    Knapp, C., & Carter, G. C. (1976).
    The generalized correlation method for estimation of time delay. 
    
    Adapted from:
    https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/experimental/localization.py
    
    Parameters
    ----------
    x1 : nd-array
        The signal of the reference microphone
    x2 : nd-array
        The signal of the second microphone
    simulation_fs : int
        The sampling frequency of the input signal
        
    Returns
    ------
    tau : float
        the delay between the two microphones (in seconds)
    '''

    # zero padded length for the FFT
    n = (x1.shape[0]+x2.shape[0]-1)
    if n % 2 != 0:
        n += 1

    X1 = fft.rfft(np.array(x1, dtype=np.float32), n=n)
    X2 = fft.rfft(np.array(x2, dtype=np.float32), n=n)

    X1 /= np.abs(X1)
    X2 /= np.abs(X2)

    cc = fft.irfft(X1*np.conj(X2), n=n)
    
    # maximum possible delay given distance between microphones
    t_max = n // 2 + 1
    #cc = np.concatenate([cc[-t_max:],cc[:t_max]])

    indxs = np.concatenate([-np.flip(np.arange(t_max)), np.arange(t_max)])

    return indxs, cc


def _gcc_phat_new(x1, x2, fs, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal x2
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT) method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(x1) + len(x2)
    n = x1.shape[0] + x2.shape[0]

    # Generalized Cross Correlation Phase Transform
    # See http://www.xavieranguera.com/phdthesis/node92.html

    X1 = np.fft.rfft(x1, n=n)
    X2 = np.fft.rfft(x2, n=n)
    R = X1 * np.conj(X2)
    Gphat = R / np.abs(R)
    cc = np.fft.irfft(Gphat, n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    
    return tau, cc
