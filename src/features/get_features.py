# Feature Extraction Functions
# Author(s): Neha Das (neha.das@tum.de)

import numpy as np
from os.path import dirname, abspath
import os, sys
import glob
import torch
import scipy.stats as sp
from scipy import signal as spsig
from scipy.signal import butter, filtfilt
import pywt

# import catch22

from src.features.wavelets import wavedec, wrcoef


def get_features_dynamic_gauss(
    X,
    sample_freq=50,
    noise_filtering_freq=np.array([0.1, 10]),
    dec_lev=9,
    window_movestd_per=0.5,
):
    """
    Creates data features given in the paper 'Dynamics based estimation of Parkinson's Disease Severity using Gaussian Processes'
    http://mediatum.ub.tum.de/doc/1454686/686005.pdf

    Args:
        X: Acceleration magnitudes of different signal sources
        sample_freq: Sample frequency
        noise_filtering_freq: Noise Filtering
        dec_lev: Wavelet decomposition level
        window_movestd_per:

    Return:
        feature set X
    """
    X_all = []
    for x in X:
        # x = x.flatten()
        # import pdb; pdb.set_trace()
        x_raw = bibutter(
            signal=x,
            sample_freq=sample_freq,
            cutoff_freq=noise_filtering_freq,
            method="bandpass",
        )
        X_all.append(x_raw)
        # x_d = getsignalD(
        #     signal=x_raw,
        #     sample_freq=self.sample_freq
        # )

        w = pywt.Wavelet("db3")
        C, L = wavedec(x_raw, wavelet=w, level=dec_lev)
        mvs = round(window_movestd_per * sample_freq)

        for n in range(dec_lev):
            sig = wrcoef(C, L, wavelet=w, level=n + 1)
            X_all.append(movestd(sig, mvs))

    X2 = []
    for x in X_all:
        rms = np.sqrt(np.nanmean(x**2, axis=-1))
        std = np.nanstd(x, axis=-1)
        max = np.nanmax(x, axis=-1)
        kur = sp.kurtosis(x, axis=-1)
        skw = sp.skew(x, axis=-1)
        freqs, psd = spsig.welch(x, fs=sample_freq, axis=-1)
        psd_max = np.nanmax(psd, axis=-1)
        psd_mean = np.nanmean(psd, axis=-1)

        # import pdb; pdb.set_trace()

        x = np.stack([rms, std, max, kur, skw, psd_max, psd_mean], axis=-1)
        X2.append(x)

    X2 = np.stack(X2)
    X2 = X2.reshape((X.shape[0], -1))

    return X2


def get_features_catch_22(X, sample_freq=50, noise_filtering_freq=np.array([0.1, 10])):
    """
    Creates data features given in the paper 'Lubba et al. (2019). catch22: CAnonical Time-series CHaracteristics.'
    https://github.com/chlubba/catch22

    Args:
        X: Acceleration magnitudes of different signal sources

    Return:
        feature set X
    """
    X_all = []
    for x in X:
        if sample_freq is not None and noise_filtering_freq is not None:
            x = bibutter(
                signal=x,
                sample_freq=sample_freq,
                cutoff_freq=noise_filtering_freq,
                method="bandpass",
            )
        X_all.append(catch22.catch22_all(x)["values"])
    X = np.stack(X_all, axis=-1)
    X = X.reshape(-1)

    return X


def rolling_window(a, window, step_size):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size + 1, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def movestd(a, window: int):
    len_a = len(a)
    # Calculated the tail ends
    start_tail = np.array([a[:i].std(ddof=1) for i in range(int(window / 2), window)])
    middle = rolling_window(a, window=10, step_size=1).std(axis=1, ddof=1)
    end_tail = np.array(
        [a[len_a - i - 1 :].std(ddof=1) for i in range(window, int(window / 2) - 1, -1)]
    )
    return np.concatenate([start_tail, middle, end_tail])


def bibutter(signal, sample_freq, cutoff_freq, method):
    n = 4
    # filter the filled data

    fnorm = cutoff_freq / (sample_freq / n)
    [b, a] = butter(n, fnorm, method)
    out = filtfilt(b, a, signal)
    return out
