import logging
from argparse import ArgumentParser
from os import mkdir, listdir
from os.path import exists, splitext, join

import scipy.io.wavfile as wav
from python_speech_features import mfcc
import numpy as np


class AudioFeaturesExtractor:

    def __init__(self):
        pass

    def process_folder(self, src_dir: str, dst_dir: str):
        if not exists(dst_dir):
            mkdir(dst_dir)

        for audio in listdir(src_dir):
            recording_name, _ = splitext(audio)
            mfccs = self.calculate_mfcc(join(src_dir, audio))
            logging.info(f"{recording_name}:{mfccs.shape}")
            np.save(join(dst_dir, recording_name + '.npy'), mfccs)

    def average(self, arr, n):
        end = n * int(len(arr) / n)
        return np.mean(arr[:end].reshape(-1, n), 1)

    def calculate_mfcc(self, audio_filename: str):
        rate, data = wav.read(audio_filename)
        # make mono from stereo
        if len(data.shape) == 2:
            data = (data[:, 0] + data[:, 1]) / 2
        mfccs = mfcc(data, winlen=0.02, winstep=0.01, samplerate=rate, numcep=26, nfft=1024)
        # average to meet to framerate (20fps)
        mfccs = [self.average(mfccs[:, i], 5) for i in range(26)]
        return np.transpose(mfccs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src_dir', help='Path to recorded speech folder')
    arg_parser.add_argument('--dst_dir', help='Path where extracted audio features will be stored')

    args = arg_parser.parse_args()
    extractor = AudioFeaturesExtractor()
    extractor.process_folder(args.src_dir, args.dst_dir)
