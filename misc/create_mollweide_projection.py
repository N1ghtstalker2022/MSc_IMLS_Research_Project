import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import argparse
import projects.deep_video_compression.utils.healpix as hp_utils
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
# Verbosity
parser.add_argument('--input', "-i", type=str, help='File address of the input healpix sampled map (stored in npy file).')
parser.add_argument('--output', "-o", type=str, help='File address indicating where to save the Mollweide projection.')
parser.add_argument('--width', '-w', type=int, help='Width of the Mollweide projection image')

def main():
    args = parser.parse_args()
    print("=========== printing args ===========")
    for key, val in args.__dict__.items():
        print("{:_<40}: {}\n".format(key, val))  # check this for all kinds of formatting
    print("=" * 60 + "\n")


    map = np.load(args.input)

    img = hp_utils.mollw(map, args.width, bilinear_interpolation=True, isNest=True)
    img = Image.fromarray(img, 'RGB')
    img.save(args.output)


if __name__ == '__main__':
    main()

