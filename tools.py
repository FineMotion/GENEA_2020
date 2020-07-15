import scipy.io.wavfile as wav
from python_speech_features import mfcc
import numpy as np


def average(arr, n):
    end = n * int(len(arr) / n)
    return np.mean(arr[:end].reshape(-1, n), 1)


def calculate_mfcc(audio_filename: str):
    rate, data = wav.read(audio_filename)
    # make mono from stereo
    if len(data.shape) == 2:
        data = (data[:, 0] + data[:, 1]) / 2
    mfccs = mfcc(data, winlen=0.02, winstep=0.01, samplerate=rate, numcep=26, nfft=1024)
    # average to meet to framerate (20fps)
    mfccs = [average(mfccs[:, i], 5) for i in range(26)]
    return np.transpose(mfccs)