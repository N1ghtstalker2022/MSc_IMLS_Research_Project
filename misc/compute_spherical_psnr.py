import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
# import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import argparse
# import cv2
from projects.deep_video_compression import utils as common_utils
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # To solve the problem on the GRID
import math
import dataset

def check_colat_valid(theta):
    """Raises exception if theta is not within 0 and pi"""
    theta = np.asarray(theta)
    if not ((theta >= 0).all() and (theta <= np.pi + 1e-5).all()):
        raise ValueError("THETA is out of range [0,pi]")

def ang2vec(colat, lon):
    """ang2vec : convert angles to 3D position vector
    """

    check_colat_valid(colat)
    sintheta = np.sin(colat)
    return sintheta * np.cos(lon), sintheta * np.sin(lon), np.cos(colat)

def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.5870, 0.1140, 0.2989])
    gray = gray.astype(rgb.dtype)
    return gray

def sampleHEALPix(hp_img, colat, lon):
    hp_img = hp_img.reshape((hp_img.shape[0], -1))
    nside = hp.npix2nside(hp_img.shape[0])

    pix, weights = hp.get_interp_weights(nside, colat, lon, nest=True, lonlat=False)
    values = np.expand_dims(weights[0], axis=1) * hp_img[pix[0], :] + \
             np.expand_dims(weights[1], axis=1) * hp_img[pix[1], :] + \
             np.expand_dims(weights[2], axis=1) * hp_img[pix[2], :] + \
             np.expand_dims(weights[3], axis=1) * hp_img[pix[3], :]

    values = np.around(values)
    assert (values >= 0).all() and (values <= 255).all()
    values = values.astype(np.uint8)
    return values

def sample_sphere(img, projection, colat, lon):
    if projection == "erp":
        samples = common_utils.sampleEquirectangular(img, colat, lon, flip=True, interpolation="bilinear")
    elif projection == "healpix":
        samples = sampleHEALPix(img, colat, lon)
    else:
        raise ValueError("Projection not defined")
    return samples

def get_imagename(file_addr, prefix="", suffix=""):
    filename = common_utils.extract_filename(file_addr)
    assert filename.startswith(prefix), "file does not start with the given prefix"
    assert filename.endswith(suffix), "file does not end with the given suffix"
    end = -len(suffix) if len(suffix) != 0 else None
    return filename[len(prefix):end]



def load_image(file_addr):
    if not os.path.isfile(file_addr):  # if original file does not exist
        raise ValueError("file " + file_addr + " does not exist")

    ext = common_utils.extract_fileExtension(file_addr)
    if ext == ".npy":
        return np.load(file_addr)

    return np.array(Image.open(file_addr))

def read_rateFile(file_addr):
    rates = dict()
    with open(file_addr) as fp:
        for line in fp:
            if line[0]=="#":
                continue
            vals = line.split()
            if vals[0] in rates:
                raise ValueError("name already exists in the rate files")

            if len(vals)==2:
                r_f = float(vals[1])
                assert math.isclose(int(round(r_f)), r_f), "Rate is in bits therefore it is an integer"
                rates[vals[0]] = {"Theoretical":None, "Actual": int(round(r_f))}
            elif len(vals)==3:
                r1_f, r2_f = float(vals[1]), float(vals[2])
                assert math.isclose(int(round(r1_f)), r1_f) and math.isclose(int(round(r2_f)), r2_f), "Rate is in bits therefore it is an integer"
                rates[vals[0]] = {"Theoretical": int(round(r1_f)), "Actual": int(round(r2_f))}
            else:
                raise ValueError("each line must have at 2 or 3 columns")
    return rates

def compute_PSNR(ref, out, weight=None):
    assert ref.dtype == out.dtype, "type must be similar"
    PIXEL_MAX = float(np.iinfo(ref.dtype).max)
    ref = ref.astype(np.float64)    # it doesn't change the type of ref outside
    out = out.astype(np.float64)    # it doesn't change the type of out outside
    mse = ref - out
    mse = mse * mse
    if weight is not None:
        assert math.isclose(weight.sum(), 1), "verify that sum is one"
        size = mse.size
        mse *= weight
        mse = mse.sum()
        if weight.size != size: # averaging channels when broadcasting happens
            a = size/weight.size
            mse /= a
    else:
        mse = np.mean(mse)
    if math.isclose(mse, 0): # for small values return zero
        return 999.99
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr

def compute_WSPSNR(ref, out, projection):
    if projection == "erp":

        height = ref.shape[0]
        weights = np.arange(0, height, dtype=np.float64)
        weights = np.cos( (weights - height/2. + 0.5) * (np.pi/height), dtype=np.float64)
        weights = np.tile(weights.reshape(-1, 1), (1, ref.shape[1]))
        weights = weights/weights.sum()
        # Handling the case if ref and out have several color channels
        s = [1,] * ref.ndim
        for i in range(weights.ndim):
            s[i] = weights.shape[i]

        wspsnr = compute_PSNR(ref, out, weights.reshape(s))
    elif projection == "healpix":
        wspsnr = compute_PSNR(ref, out)
    else:
        raise ValueError("Projection not defined")
    return wspsnr
parser = argparse.ArgumentParser()

parser.add_argument("--original-dir", "-o", type=str, help="Directory where the original images are.")
parser.add_argument("--original-ext", "-oe", type=str, choices=["jpg", "jpeg", "png", "tiff", "npy"], help="Extension of the original images.")
parser.add_argument("--original-prefix", "-op", type=str, default="", help="Prefix in the of the original file names.")
parser.add_argument("--original-suffix", "-os", type=str, default="", help="Suffix in the of the original file names.")
parser.add_argument("--projection-original", "-po", type=str.lower, choices=["healpix", "erp"], help="Projection type.")

parser.add_argument('--test-images_equirectangular', '-t', type=str, help="A text file containing location of test images.")
parser.add_argument("--test-files-prefix", type=str, default="", help="Common prefix in the uncompressed test file names.")
parser.add_argument("--test-files-suffix", type=str, default="", help="Common suffix in the uncompressed test file names.")

parser.add_argument("--reconstruction-dir", "-r", type=str, help="Main directory where the reconstructed projections are.")
parser.add_argument("--reconstruction-subfolder", "-rf", type=str, help="Folder name where the reconstructed projections are.")
parser.add_argument("--reconstruction-ext", "-re", type=str, choices=["jpg", "jpeg", "png", "tiff", "npy"], help="Extension of the reconstructed projections.")
parser.add_argument("--reconstruction-prefix", "-rp", type=str, default="", help="Common prefix in the reconstructed file names.")
parser.add_argument("--reconstruction-suffix", "-rs", type=str, default="", help="Common suffix in the reconstructed file names.")

parser.add_argument("--projection", "-p", type=str.lower, choices=["healpix", "erp"], help="Projection type.")

parser.add_argument("--rate-prefix", "-ratef", type=str, default="", help="File name prefix of each reconstructed image in the rate text file.")
parser.add_argument("--rate-suffix", "-rates", type=str, default="", help="File name suffix in each reconstructed image in the rate text file.")
parser.add_argument("--sphere-points", "-s", type=str, help="File that shows sphere points uniformly sampled.")


def main(argv):
    args = parser.parse_args(argv)
    print("=========== printing args ===========")
    for key, val in args.__dict__.items():
        print("{:_<40}: {}\n".format(key, val))  # check this for all kinds of formatting
    print("=" * 60 + "\n")


    nameWidth = 20
    psnrWidth = 22
    rateWidth = 30

    colat_lon = np.loadtxt(args.sphere_points, dtype=np.float64)
    colat_lon += np.array([90, 180], dtype=colat_lon.dtype)
    colat_lon = np.radians(colat_lon)
    colat, lon = colat_lon[:, 0], colat_lon[:, 1]

    test_dataset = dataset.HealpixDataset(args.test_data)

    rates_dict = read_rateFile(os.path.join(args.reconstruction_dir, args.reconstruction_subfolder, "rates.txt"))

    SpsnrMeter_color = common_utils.AverageMeter('Spsnr color', ':.2f')
    SpsnrMeter_gray = common_utils.AverageMeter('Spsnr gray', ':.2f')
    WSpsnrMeter_color = common_utils.AverageMeter('WSpsnr color', ':.2f')
    WSpsnrMeter_gray = common_utils.AverageMeter('WSpsnr gray', ':.2f')
    actualRateMeter = common_utils.AverageMeter('Actual rate', ':d')
    theoreticalRateMeter = common_utils.AverageMeter('Theoretical rate', ':d')

    with open(os.path.join(args.reconstruction_dir, "rate_spherical_psnrs_"+ args.reconstruction_subfolder +".txt"), 'w') as f:
        fmt_psnr = '{:^' + str(psnrWidth) + 's}'
        fmt_rate = '{:^' + str(rateWidth) + 's}'
        header = "#" + "name".center(nameWidth - 1) + \
                 (fmt_psnr * 4).format("SPSNR Gray", "WSPSNR Gray", "SPSNR Color", "WSPSNR Color") +\
                 (fmt_rate * 2).format("Theoretical Rate", "Actual Rate")

        f.write(header+"\n")

        header_units = "#" + " " * (nameWidth - 1) + \
                 (fmt_psnr * 4).format("(dB)", "(dB)", "(dB)", "(dB)") + \
                 (fmt_rate * 2).format("(bits)", "(bits)")

        f.write(header_units+"\n")

        fmt_psnr = '{:^' + str(psnrWidth) + '.2f}'
        fmt_rate = '{:^' + str(rateWidth) + 'd}'

        for i, uncompressed_filename in enumerate(test_dataset.list_filenames):
        
            image_name = get_imagename(uncompressed_filename, args.test_files_prefix, args.test_files_suffix)
            print("Analyzing image {} [{:04n}/{:04n}]".format(image_name, i, len(test_dataset.list_filenames)), flush=True)

            orig_addr = os.path.join(args.original_dir, args.original_prefix+image_name+args.original_suffix+"."+args.original_ext)
            orig_img = load_image(orig_addr)

            rec_addr = os.path.join(args.reconstruction_dir, args.reconstruction_subfolder, args.reconstruction_prefix+image_name+args.reconstruction_suffix+"."+args.reconstruction_ext)
            rec_img = load_image(rec_addr)

            # uncompressed_addr = os.path.join(test_dataset.root_dir, args.uncompressed_prefix+image_name+args.uncompressed_suffix+"."+args.uncompressed_ext)
            uncompressed_addr = os.path.join(test_dataset.root_dir, uncompressed_filename)
            uncompressed_img = load_image(uncompressed_addr)
            ### Compute SPSNR
            ref_sampleSPSNR = sample_sphere(orig_img, args.projection_original, colat, lon)
            out_sampleSPSNR = sample_sphere(rec_img, args.projection, colat, lon)

            SPSNR = compute_PSNR(ref_sampleSPSNR, out_sampleSPSNR)
            SpsnrMeter_color.update(SPSNR)
            SPSNR_gray = compute_PSNR(rgb2gray(ref_sampleSPSNR), rgb2gray(out_sampleSPSNR))
            SpsnrMeter_gray.update(SPSNR_gray)


            ### Compute WSPSNR
            WSPSNR = compute_WSPSNR(uncompressed_img, rec_img, args.projection)
            WSpsnrMeter_color.update(WSPSNR)
            WSPSNR_gray = compute_WSPSNR(rgb2gray(uncompressed_img), rgb2gray(rec_img), args.projection)
            WSpsnrMeter_gray.update(WSPSNR_gray)

            rate = rates_dict.get(args.rate_prefix+image_name+args.rate_suffix, None)
            if rate is None:
                raise ValueError("The image does not exist in rate file")

            rate_theoretical = rate.get("Theoretical", None)
            if rate_theoretical is None:    # -1 because In classical compressions we do not have access to Theoretical rates
                rate_theoretical = -1
            theoreticalRateMeter.update(rate_theoretical)

            rate_actual = rate.get("Actual", None)
            actualRateMeter.update(rate_actual)


            line = image_name.center(nameWidth) + \
                   (fmt_psnr * 4).format(SPSNR_gray, WSPSNR_gray, SPSNR, WSPSNR) + \
                   (fmt_rate * 2).format(rate_theoretical, rate_actual)

            f.write(line+"\n")

        # Printing the average values
        fmt_rate = '{:^' + str(rateWidth) + '.2f}'
        line = "Average".center(nameWidth) + \
               (fmt_psnr * 4).format(SpsnrMeter_gray.avg, WSpsnrMeter_gray.avg, SpsnrMeter_color.avg, WSpsnrMeter_color.avg) + \
               (fmt_rate * 2).format(theoreticalRateMeter.avg, actualRateMeter.avg)
        f.write("-"*len(line)+"\n") # before that print a line of "-"
        f.write(line + "\n")



    print("finished")

if __name__ == "__main__":
    main(sys.argv[1:])

