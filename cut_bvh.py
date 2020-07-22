from pymo.parsers import BVHParser
from pymo import viz_tools
from pymo.preprocessing import MocapParameterizer
from pymo.writers import BVHWriter
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src', help='Source BHV file')
    parser.add_argument('--dst', help='Result BVH file path')
    args = parser.parse_args()

    bvh_parser = BVHParser()
    data = bvh_parser.parse(args.src)

    target_fps = 20
    orig_fps = round(1.0 / data.framerate)
    rate = orig_fps // target_fps
    print(orig_fps, rate)

    new_data = data.clone()
    new_data.values = data.values[0:-1:rate]
    new_data.values = new_data.values[:1200]
    new_data.framerate = 1.0 / target_fps

    bvh_writer = BVHWriter()
    with open(args.dst, 'w') as f:
        bvh_writer.write(new_data, f)
