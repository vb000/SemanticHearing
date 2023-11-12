import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import rfft, irfft
from scipy.signal import stft
from pyroomacoustics.doa import srp
from pyroomacoustics.experimental.localization import tdoa
import pyroomacoustics as pra
import src.helpers.utils as utils
import torch

try:
    import mklfft as fft
except ImportError:
    import numpy.fft as fft


def tdoa2(x1, x2, interp=1, fs=1, phat=True, t_max=None):
    """
    This function computes the time difference of arrival (TDOA)
    of the signal at the two microphones. This in turns is used to infer
    the direction of arrival (DOA) of the signal.
    Specifically if s(k) is the signal at the reference microphone and
    s_2(k) at the second microphone, then for signal arriving with DOA
    theta we have
    s_2(k) = s(k - tau)
    with
    tau = fs*d*sin(theta)/c
    where d is the distance between the two microphones and c the speed of sound.
    We recover tau using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)
    method. The reference is
    Knapp, C., & Carter, G. C. (1976). The generalized correlation method for estimation of time delay.
    Parameters
    ----------
    x1 : nd-array
        The signal of the reference microphone
    x2 : nd-array
        The signal of the second microphone
    interp : int, optional (default 1)
        The interpolation value for the cross-correlation, it can
        improve the time resolution (and hence DOA resolution)
    fs : int, optional (default 44100 Hz)
        The sampling frequency of the input signal
    Return
    ------
    theta : float
        the angle of arrival (in radian (I think))
    pwr : float
        the magnitude of the maximum cross correlation coefficient
    delay : float
        the delay between the two microphones (in seconds)
    """
    # zero padded length for the FFT
    n = x1.shape[-1] + x2.shape[-1] - 1
    if n % 2 != 0:
        n += 1

    # Generalized Cross Correlation Phase Transform
    # Used to find the delay between the two microphones
    # up to line 71
    X1 = fft.rfft(np.array(x1, dtype=np.float32), n=n, axis=-1)
    X2 = fft.rfft(np.array(x2, dtype=np.float32), n=n, axis=-1)

    if phat:
        X1 /= np.abs(X1)
        X2 /= np.abs(X2)

    cc = fft.irfft(X1 * np.conj(X2), n=interp * n, axis=-1)

    # maximum possible delay given distance between microphones
    
    if t_max is None:
        t_max = n // 2 + 1

    # reorder the cross-correlation coefficients
    cc = np.concatenate((cc[..., -t_max:], cc[..., :t_max]), axis=-1)

    # import matplotlib.pyplot as plt
    
    # t = np.arange(-t_max/fs, (t_max)/fs, 1/fs) * 1e6
    # plt.plot(t, cc[15])
    # plt.show()

    # pick max cross correlation index as delay
    tau = np.argmax(np.abs(cc), axis=-1)
    tau -= t_max  # because zero time is at the center of the array

    return tau / (fs * interp)


from sklearn.utils.extmath import weighted_mode
def framewise_gccphat(x, frame_dur, sr, window='tukey'):
    TMAX = int(round(1e-3 * sr))
    frame_width = int(round(frame_dur * sr))
    
    # Total number of frames
    T = 1 + (x.shape[-1] - 1)// frame_width
    
    # Drop samples to get a multiple of frame size
    if x.shape[-1] % T != 0:
        x = x[..., -x.shape[-1]%T:]
    
    assert x.shape[-1] % T == 0
    frames = np.array(np.split(x, T, axis=-1))

    window = signal.get_window(window, frame_width)
    frames = frames * window

    # Consider only frames that have energy above some threshold (ignore silence)
    ENERGY_THRESHOLD = 5e-4
    frame_energy = np.max(np.mean(frames**2, axis=-1)**0.5, axis=-1)
    mask = frame_energy > ENERGY_THRESHOLD
    frames = frames[mask]
    
    fw_gccphat = tdoa2(frames[..., 0, :], frames[..., 1, :], fs=sr, t_max=TMAX)
    
    # print(mask)
    # print(fw_gccphat)
    # print(frame_energy[mask])
    itd = weighted_mode(fw_gccphat, frame_energy[mask], axis=-1)[0]
    return itd[0]

def fw_itd_diff(s_est, s_gt, sr, frame_duration=0.25):
    """
    Computes frame-wise delta ITD
    """
    # print("GT")
    itd_gt = framewise_gccphat(s_gt, frame_duration, sr) * 1e6
    # print("GT FW_ITD", itd_gt)
    # print("EST")
    itd_est = framewise_gccphat(s_est, frame_duration, sr) * 1e6
    # print("EST FW_ITD", itd_est)
    return np.abs(itd_est - itd_gt)

def cal_interaural_error(predictions, targets, sr, debug=False):
    """Compute ITD and ILD errors
    input: (1, time, channel, speaker)
    """
    
    TMAX = int(round(1e-3 * sr))
    EPS = 1e-8
    s_target = targets[0]  # [T,E,C]
    s_prediction = predictions[0]  # [T,E,C]

    # ITD is computed with generalized cross-correlation phase transform (GCC-PHAT)
    ITD_target = [
        tdoa2(
            s_target[:, 0, i].cpu().numpy(),
            s_target[:, 1, i].cpu().numpy(),
            fs=sr,
            t_max=TMAX
        )
        * 10 ** 6
        for i in range(s_target.shape[-1])
    ]
    if debug:
        print("TARGET ITD", ITD_target)
    
    ITD_prediction = [
        tdoa2(
            s_prediction[:, 0, i].cpu().numpy(),
            s_prediction[:, 1, i].cpu().numpy(),
            fs=sr,
            t_max=TMAX,
        )
        * 10 ** 6
        for i in range(s_prediction.shape[-1])
    ]
    
    if debug:
        print("PREDICTED ITD", ITD_prediction)
    
    ITD_error1 = np.mean(
        np.abs(np.array(ITD_target) - np.array(ITD_prediction))
    )
    ITD_error2 = np.mean(
        np.abs(np.array(ITD_target) - np.array(ITD_prediction)[::-1])
    )
    ITD_error = min(ITD_error1, ITD_error2)

    # ILD  = 10 * log_10(||s_left||^2 / ||s_right||^2)
    ILD_target_beforelog = torch.sum(s_target[:, 0] ** 2, dim=0) / (
        torch.sum(s_target[:, 1] ** 2, dim=0) + EPS
    )
    ILD_target = 10 * torch.log10(ILD_target_beforelog + EPS)  # [C]
    ILD_prediction_beforelog = torch.sum(s_prediction[:, 0] ** 2, dim=0) / (
        torch.sum(s_prediction[:, 1] ** 2, dim=0) + EPS
    )
    ILD_prediction = 10 * torch.log10(ILD_prediction_beforelog + EPS)  # [C]

    ILD_error1 = torch.mean(torch.abs(ILD_target - ILD_prediction))
    ILD_error2 = torch.mean(torch.abs(ILD_target - ILD_prediction.flip(0)))
    ILD_error = min(ILD_error1.item(), ILD_error2.item())

    return ITD_error, ILD_error

def compute_itd(s_left, s_right, sr, t_max = None):
    corr = signal.correlate(s_left, s_right)
    lags = signal.correlation_lags(len(s_left), len(s_right))
    corr /= np.max(corr)

    mid = len(corr)//2 + 1
        
    # print(corr[-t_max:])
    cc = np.concatenate((corr[-mid:], corr[:mid]))

    if t_max is not None:
    # if False:
        # print(cc[-t_max:].shape)
        cc = np.concatenate([cc[-t_max+1:], cc[:t_max+1]])
    else:
        t_max = mid

    # print("OKKK", cc.shape)
    # t = np.arange(-t_max/sr, (t_max)/sr, 1/sr) * 1e6
    # plt.plot(t, np.abs(cc))
    # plt.show()
    tau = np.argmax(np.abs(cc))
    tau -= t_max
    # tau = lags[x]
    # print(tau/ sr * 1e6)

    return tau / sr * 1e6


def compute_doa(mic_pos, s, sr, nfft=2048, num_sources=1):
    # freq_range = [100, 20000]
    
    X = pra.transform.stft.analysis(s.T, nfft, nfft // 2, )
    X = X.transpose([2, 1, 0])
    
    algo_names = ['SRP', 'MUSIC', 'FRIDA', 'TOPS', 'WAVES', 'CSSM', 'NormMUSIC']
    
    srp = pra.doa.algorithms['NormMUSIC'](mic_pos.T, sr, nfft, c=343, num_sources=num_sources)
    srp.locate_sources(X)
    
    values = srp.grid.values
    phi = np.linspace(-np.pi, np.pi, 360)

    values = np.roll(values, shift=180)

    # plt.plot(phi * 180 / np.pi, values)
    # plt.xlim([-90, 90])
    # plt.show()

    peak_idx = 90 + np.argmax(values[90:270])
    return phi[peak_idx]

def doa_diff(mic_pos, est, gt, sr):
    doa_est = compute_doa(mic_pos, est, sr)
    doa_gt = compute_doa(mic_pos, gt, sr)
    return np.abs(doa_gt - doa_est)

def gcc_phat(s_left, s_right, sr):
    X = rfft(s_left)
    Y = rfft(s_right)

    Z = X * np.conj(Y)

    y = irfft(np.exp(1j * np.angle(Z)))
    center = (len(y) + 1)//2
    y = np.concatenate([y[center:], y[:center]])
    lags = (np.linspace(0, len(y), len(y)) - ((len(y) + 1) / 2)) / sr
    x = np.argmax(y)
    tau = lags[x]

    return lags, y

def compute_ild(s_left, s_right):
    sum_sq_left = np.sum(s_left ** 2, axis=-1)
    sum_sq_right = np.sum(s_right ** 2, axis=-1)
    # print(sum_sq_left)
    # print(sum_sq_right)
    return 10 * np.log10(sum_sq_left / sum_sq_right)

def itd_diff(s_est, s_gt, sr):
    """
    Computes the ITD error between model estimate and ground truth
    input: (*, 2, T), (*, 2, T)
    """
    TMAX = int(round(1e-3 * sr))
    itd_est = compute_itd(s_est[..., 0, :], s_est[..., 1, :], sr, TMAX)
    itd_gt = compute_itd(s_gt[..., 0, :], s_gt[..., 1, :], sr, TMAX)
    return np.abs(itd_est - itd_gt)

def gcc_phat_diff(s_est, s_gt, sr):
    TMAX = int(round(1e-3 * sr))
    itd_est = tdoa2(s_est[..., 0, :], s_est[..., 1, :], fs=sr, t_max=TMAX)
    itd_gt = tdoa2(s_gt[..., 0, :], s_gt[..., 1, :], fs=sr, t_max=TMAX)
    return np.abs(itd_est - itd_gt) * 10 ** 6

def ild_diff(s_est, s_gt):
    """
    Computes the ILD error between model estimate and ground truth
    input: (*, 2, T), (*, 2, T)
    """
    ild_est = compute_ild(s_est[..., 0, :], s_est[..., 1, :])
    ild_gt = compute_ild(s_gt[..., 0, :], s_gt[..., 1, :])
    return np.abs(ild_est - ild_gt)

def si_sdr(estimated_signal, reference_signals, scaling=True):
    """
    This is a scale invariant SDR. See https://arxiv.org/pdf/1811.02508.pdf
    or https://github.com/sigsep/bsseval/issues/3 for the motivation and
    explanation
    Input:
        estimated_signal and reference signals are (N,) numpy arrays
    Returns: SI-SDR as scalar
    """

    Rss = np.dot(reference_signals, reference_signals)
    this_s = reference_signals

    if scaling:
        # get the scaling factor for clean sources
        a = np.dot(this_s, estimated_signal) / Rss
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    SDR = 10 * np.log10(Sss/Snn)

    return SDR


if __name__ == "__main__":
    fs = 44100
    
    corners = np.array([[-2, 2],
                        [2, 2],
                        [2, -2],
                        [-2, -2]]).T

    room = pra.room.Room.from_corners(corners,
                                    absorption=1,
                                    fs=fs,
                                    max_order=1)
    
    # x = utils.read_audio_file('outputs/bin_gt.wav', fs)
    x_gt = utils.read_audio_file('save_examples_few/00622/gt.wav', fs)
    x_est = utils.read_audio_file('save_examples_few/00622/binaural.wav', fs)

    # framewise_gccphat(x, 0.25, fs)
    print(fw_itd_diff(x_est, x_gt, fs))
    # x = utils.read_audio_file('save_examples_val/00000/gt.wav', fs)
    # y = utils.read_audio_file('tests/sample_audio2.wav', fs)
    # mic_positions = np.array([[0, 0.09], [0, -0.09]])
    # room.add_microphone_array(mic_positions.T)

    # a1 = 50 * np.pi / 180
    # a2 = 60 * np.pi / 180
    
    # s1 = np.array([np.cos(a1), np.sin(a1)])
    # room.add_source(s1, signal=x)

    # # s2 = np.array([np.cos(a2), np.sin(a2)])
    # # room.add_source(s2, signal=y)

    # room.simulate()
    
    # s = room.mic_array.signals # (M, T)
    # s = s.transpose() # (T, M)
    # s = np.reshape(s, (1, *s.shape, 1))

    # s_est = s.copy() + np.random.normal(0, 1e-2, s.shape)
    # s_est[0, :, 0, 0] = np.roll(s_est[0, : , 0, 0], shift=222)

    # s = torch.from_numpy(s)
    # s_est = torch.from_numpy(s_est)

    # # itd_error, ild_error = cal_interaural_error(s_est, s, fs)
    # # print('ITD', itd_error)
    # # print('ILD', ild_error)

    # itd_error = itd_diff(s_est, s, fs)
    # print('ITD', itd_error)
    
    # doa = compute_doa(mic_positions, s, fs, num_sources=2)
    # print(doa * 180 / np.pi)

    # x = np.array([x[0], x[0]])
    # x[0] = np.roll(x[0], shift=2) * 0.5
    # # np.random.seed(0)
    # x = x + np.random.normal(loc=0, scale=1e-2, size=x.shape)

    # x = x[:, 140000:140000 + 190000] 
    # x = x

    # fig, ax = plt.subplots()
    # ax.plot(x[0])
    # ax.plot(x[1])

    # tdoa2(x[0, :], x[1, :], fs=fs, t_max=44)
    # utils.write_audio_file('gcc.wav', x, fs)

    # tau = compute_itd(x, y, 44100)
    # # print(tau)
    # lags, z = gcc_phat(x, y, 44100)
    # plt.plot(t, x)
    # plt.plot(t, y)
    # plt.plot(lags, z)
    # plt.grid()
    # plt.show()
