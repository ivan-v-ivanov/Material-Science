import numpy as np
import scipy.stats as st
import os
from scipy.ndimage.filters import gaussian_filter


def make_filelist(loaddir, checking_string, type_of_name_check='end'):
    filelist = []
    if type_of_name_check == 'end':
        for file in os.listdir(loaddir):
            if file.endswith('.%s' % checking_string):
                filelist.append(file)
    elif type_of_name_check == 'start':
        for file in os.listdir(loaddir):
            if file.startswith('%s.' % checking_string):
                filelist.append(file)
    filelist.sort()
    return filelist


def make_directorylist(loaddir):
    directorylist = [path[0][path[0].rfind('/') + 1:] for path in os.walk(loaddir)]
    directorylist.sort()
    directorylist = directorylist[:-1]
    return directorylist


def mean_confidence_interval(values, alpha=0.9):
    n = len(values)
    average = np.mean(values)
    standart_deviation = st.sem(values)

    if n < 30:
        interval = st.t.interval(alpha, df=n - 1, loc=average, scale=standart_deviation)
    else:
        interval = st.norm.interval(alpha, loc=average, scale=standart_deviation)

    delta = interval[1] - average
    return average, delta


def find_maxima(signal, sigma, lower_limit):
    norm_signal = signal / np.max(signal)
    filtered_signal = gaussian_filter(norm_signal, sigma)
    ind_max = (np.diff(np.sign(np.diff(filtered_signal))) < 0).nonzero()[0] + 1
    ind_max = [ind for ind in ind_max if filtered_signal[ind] > lower_limit]
    number_of_maxima = len(ind_max)
    return number_of_maxima, ind_max


def find_closest(values, target):
    idx = np.searchsorted(values, target)
    idx = np.clip(idx, 1, len(values) - 1)
    left = values[idx - 1]
    right = values[idx]
    idx -= target - left < right - target
    return idx


# sensetive to nans
# def find_closest(values, target):
#     return np.argmin(np.abs([val - target for val in values]))