from argparse import ArgumentParser
import logging
from os import listdir, mkdir
from os.path import join, exists, split
import numpy as np

from audio_utils import calculate_mfcc


class DataAligner:
    def __init__(self, args):
        self.motion_dir = args.motion_dir
        self.audio_dir = args.audio_dir
        self.with_context = args.with_context
        self.context_length = args.context_length

    @staticmethod
    def shorten(arr1, arr2):
        min_len = min(len(arr1), len(arr2))
        arr1 = arr1[:min_len]
        arr2 = arr2[:min_len]
        return arr1, arr2

    def pad_audio(self, audio_data):
        silence = calculate_mfcc("silence.wav")
        paddings = np.array([silence[0]] * (self.context_length // 2))
        result = np.append(paddings, audio_data, axis=0)
        result = np.append(result, paddings, axis=0)
        return result

    def contextualize(self, audio_data: np.ndarray):
        strides = len(audio_data)
        audio_data = self.pad_audio(audio_data)
        logging.debug(f"Padded audio shape: {audio_data.shape}")
        audio_with_context = []
        # audio_data[0: self.context_length + 1].reshape(1, self.context_length + 1, -1)
        logging.debug(f"Strides: {strides}")
        for i in range(strides):
            audio_with_context.append(audio_data[i:i + self.context_length + 1])
        audio_with_context = np.array(audio_with_context)
        logging.debug(f"Output audio shape: {audio_with_context.shape}")
        return audio_with_context

    def align_recording(self, audio_file: str, motion_file: str):
        assert exists(audio_file) and exists(motion_file)
        audio_data = np.load(audio_file)
        motion_data = np.load(motion_file)
        logging.debug(f"Input audio shape: {audio_data.shape}\tInput motion shape: {motion_data.shape}")

        audio_data, motion_data = self.shorten(audio_data, motion_data)
        logging.debug(f"Shorten audio shape: {audio_data.shape}\tShorten motion shape: {motion_data.shape}")
        if not self.with_context:
            return audio_data, motion_data

        # convert audio data with context
        audio_with_context = self.contextualize(audio_data)

        return audio_with_context, motion_data

    def align(self, dst_dir: str):
        if not exists(dst_dir):
            mkdir(dst_dir)
        logging.info('Aligning data...')
        for i, recording in enumerate(listdir(self.audio_dir)):
            logging.info(recording)
            # audio and motion files have the same names
            audio_file = join(self.audio_dir, recording)
            if self.motion_dir is not None:
                motion_file = join(self.motion_dir, recording)
                audio_data, motion_data = self.align_recording(audio_file, motion_file)
                np.savez(join(dst_dir, "data_%03d.npz" % (i + 1)), X=audio_data, Y=motion_data)
            else:
                audio_data = self.contextualize(np.load(audio_file))
                print(audio_data.shape)
                np.save(join(dst_dir, split(audio_file)[-1]), audio_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--motion_dir', type=str, default=None, help='Path to motion features folder')
    arg_parser.add_argument('--audio_dir', help='Path to audio features folder')
    arg_parser.add_argument('--dst_dir', help='Path where aligned data will be stored')
    arg_parser.add_argument('--with_context', action="store_true", help='Set use audio data with context or not')
    arg_parser.add_argument('--context_length', help='Length of the speech context', type=int, default=60)
    args = arg_parser.parse_args()

    aligner = DataAligner(args)
    aligner.align(args.dst_dir)
