import numpy as np
from numpy.fft import fft, ifft

def cal_period(time_series: np.ndarray, sampling_rate=1):
    time_series_in_freq = fft(time_series)
    sample_num = len(time_series)
    sample_time = sample_num / sampling_rate

    time_series_in_freq[0] = 0
    main_freq = np.argmax(time_series_in_freq)
    p = round(sample_time / main_freq)

    if 3 * p > sample_num:
        p = 1
    return p


def mse_obj(x_truth: np.ndarray, x_estimated: np.ndarray):
    assert x_truth.shape == x_estimated.shape

    return np.average(np.square(x_truth - x_estimated))


def POF(x_t_back, x_t, x_t_forward):
    eps = 1

    def d(x1, x2):
        return np.sqrt(np.sum(np.square(x1 - x2)))
    
    d_xt_xt_b = d(x_t, x_t_back)
    d_xt_xt_f = d(x_t, x_t_forward)
    d_xt_b_xt_f = d(x_t_back, x_t_forward)

    y1 = (d_xt_xt_b + d_xt_xt_f) / (d_xt_xt_b + d_xt_b_xt_f + eps)
    y2 = (d_xt_xt_b + d_xt_xt_f) / (d_xt_xt_f + d_xt_b_xt_f + eps)

    pof = (y1 + y2) / 2

    if np.isnan(pof):
        pof = 2

    return pof
    

def nf_obj(x_estimated: np.ndarray):
    p = cal_period(x_estimated)

    pos_ls = []
    for pos in range(p, len(x_estimated)- 2*p + 1):
        x_t_back = x_estimated[pos-p : pos]
        x_t = x_estimated[pos : pos+p]
        x_t_forward = x_estimated[pos+p : pos+2*p]

        pos_ls.append(POF(x_t_back, x_t, x_t_forward))

    return np.average(pos_ls)


def mse_nf(x_truth: np.ndarray, x_estimated: np.ndarray):
    return mse_obj(x_truth, x_estimated) + nf_obj(x_estimated)