from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import join, splitext, exists

import numpy as np
from sklearn.pipeline import Pipeline

from pymo.parsers import BVHParser
from pymo.preprocessing import DownSampler, RootTransformer, JointSelector, MocapParameterizer, ConstantsRemover, \
    Numpyfier
import logging
import joblib as jl


class MotionFeaturesExtractor:
    def __init__(self):
        pass

    def process_folder(self, src_dir: str, dst_dir: str, pipeline_dir, fps: int = 20):
        bvh_parser = BVHParser()
        data = []
        bvh_names = listdir(src_dir)
        logging.info('Parsing BVH files...')
        for bvh_name in bvh_names:
            bvh_path = join(src_dir, bvh_name)
            logging.info(bvh_path)
            data.append(bvh_parser.parse(bvh_path))

        # pipeline from https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=fps, keep_all=False)),
            ('root', RootTransformer('hip_centric')),
            # ('mir', Mirror(axis='X', append=True)),
            ('jtsel', JointSelector(
                ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm',
                 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'],
                include_root=True)),
            ('exp', MocapParameterizer('expmap')),
            ('cnst', ConstantsRemover()),
            ('np', Numpyfier())
        ])
        logging.info('Transforming data...')
        out_data = data_pipe.fit_transform(data)
        if not exists(pipeline_dir):
            mkdir(pipeline_dir)
        jl.dump(data_pipe, join(pipeline_dir, 'data_pipe.sav'))

        logging.info('Saving result...')
        if not exists(dst_dir):
            mkdir(dst_dir)
        for i, bvh_name in enumerate(bvh_names):
            name, _ = splitext(bvh_name)
            logging.info(name)
            np.save(join(dst_dir, name + ".npy"), out_data[i])
            # np.savez(join(dst_dir, name + "_mirrored.npz"), clips=out_data[len(bvh_names) + i])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src_dir', help='Path to original motions folder')
    arg_parser.add_argument('--dst_dir', help='Path where extracted features will be stored')
    arg_parser.add_argument('--pipeline_dir', default='./pipe', help='Path to save pipeline')

    args = arg_parser.parse_args()
    extractor = MotionFeaturesExtractor()
    extractor.process_folder(args.src_dir, args.dst_dir, args.pipeline_dir)

