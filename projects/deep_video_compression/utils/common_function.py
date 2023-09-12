import ntpath
import os
import numpy as np
from collections import defaultdict
import cv2
import torch
from PIL import Image
import re
import pirt

def is_power2(num):
    """states if a number is a power of two"""
    return np.logical_and(num != 0, (num & (num - 1)) == 0)


def vectorize_with_zigzag_order(a):
    return np.concatenate([np.diagonal(a[::-1, :], k)[::(2 * (k % 2) - 1)] for k in range(1 - a.shape[0], a.shape[0])])


def scale_range(input, output_min, output_max):
    input += -(np.min(input))
    input /= np.max(input) / (output_max - output_min)
    input += output_min
    return input


def extract_filename(path):
    head, tail = ntpath.split(path)
    base = tail or ntpath.basename(head)
    return os.path.splitext(base)[0]

def extract_fileExtension(path):
    filename, file_extension = os.path.splitext(path)
    return file_extension


def checkConsecutive(l):
    # Check if list contains consecutive numbers
    return sorted(l) == list(range(min(l), max(l)+1))


def get_indexesDuplicateItems(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return dict([(key, locs) for key, locs in tally.items()])


def imread(path, dsize=None):
    # Reads (and, optionally, resizes) an image.
    I = Image.open(path)

    if dsize is not None:
        I = I.resize(dsize)

    return np.array(I)


def sumOfAP(a, d, n): # sum of Arithmetic Progression (forced to return an integer by // in //2)
        return (n * (2 * a + (n - 1) * d)) // 2


def remap(value, maxInput, minInput, maxOutput, minOutput):
    value = np.clip(value, minInput, maxInput)

    inputSpan = maxInput - minInput
    outputSpan = maxOutput - minOutput

    scaledThrust = (value - float(minInput)) / float(inputSpan)

    return minOutput + (scaledThrust * outputSpan)


def sampleEquirectangular(equi_img, colat, lon, flip, interpolation):
    assert colat.shape == lon.shape, "Shapes must be similar"
    assert interpolation in ["bilinear", "lanczos"], "Interpolation not defined"

    if flip:
        equi_img = torch.flip(equi_img, [2])  # Flipping along width for batch

    y = (colat / np.pi) * equi_img.shape[1]
    lon[lon < 0.] += 2 * np.pi
    lon[lon > 2 * np.pi] -= 2 * np.pi
    x = (lon / (2 * np.pi)) * equi_img.shape[2]

    if interpolation == "bilinear":
        y = y - 0.5
        y = torch.clamp(y, 0., equi_img.shape[1] - 1.)
        x = x - 0.5
        x[x < 0] += equi_img.shape[2]

        i0 = y.long()
        j0 = x.long()
        i1 = i0 + 1
        j1 = (j0 + 1) % equi_img.shape[2]

        i0 = torch.clamp(i0, 0, equi_img.shape[1] - 1)
        j0 = torch.clamp(j0, 0, equi_img.shape[2] - 1)
        i1 = torch.clamp(i1, 0, equi_img.shape[1] - 1)
        j1 = torch.clamp(j1, 0, equi_img.shape[2] - 1)

        frac_x = y - i0.float()
        frac_y = x - j0.float()

        # Reshape for broadcasting
        frac_x = frac_x.view(-1, 1)
        frac_y = frac_y.view(-1, 1)

        # Explicitly broadcasting the tensors for batch processing
        values = (1 - frac_x) * (1 - frac_y) * equi_img[:, i0, j0, :] + \
                 frac_x * (1 - frac_y) * equi_img[:, i1, j0, :] + \
                 (1 - frac_x) * frac_y * equi_img[:, i0, j1, :] + \
                 frac_x * frac_y * equi_img[:, i1, j1, :]

        values = values.type(equi_img.dtype)
        return values.view(equi_img.shape[0], *colat.shape, -1)
    else:
        raise NotImplementedError("Lanczos interpolation is not yet implemented.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, batch_mean, n=1):
        self.val = batch_mean
        self.sum += float(batch_mean * n)
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def read_RDfile(file_addr, deleteAverageField):

    def replace_brackets_and_avg(line):
        # replace any kind of content between brackets "(\[(.*?)\])" and if there is zero or one word "Avg" before brackets with an empty space
        return re.sub(r'(Avg)?\[(.*?)\]', " ", line)

    assert os.path.isfile(file_addr), "File does not exist"
    with open(file_addr) as fp:
        first_line = replace_brackets_and_avg(fp.readline())
        list_header = re.split(r'\s{2,}', first_line) # split a string with at least 2 whitespaces
        assert list_header[0] == "#", "The first line must start with #"
        assert list_header[1] == "name", "The first line first element must be name"
        list_header = list(filter(lambda x: x not in ["", "#", "name"], list_header))   # Remove "", "#", "name" from the string
        # print(list_header)

        second_line = replace_brackets_and_avg(fp.readline())
        list_units = re.split(r'\s{2,}', second_line) # split a string with at least 2 whitespaces
        assert list_units[0] == "#", "The second line must start with #"
        list_units = list(filter(lambda x: x not in ["", "#"], list_units))  # Remove "" and "#" from the string
        # print(list_units)


        assert len(list_header) == len(list_units), "list_header must have one more element than list units (name)"

        dict_rd = dict()
        for line in fp:
            if all(elem == "-" for elem in line[:-1]): # Arriving to the last element,
                line = fp.readline() # skip this line and get the next line which are the average values

            line = replace_brackets_and_avg(line)
            vals = line.split()
            img_name = vals[0]
            vals = vals[1:]

            assert len(vals) == len(list_header), "number of elements in line must correspond to number of elements in the header"
            if img_name in dict_rd:
                print(img_name)
                raise ValueError("name already exists in the dictionary")
            dict_rd[img_name] = {list_header[i] : float(vals[i]) for i in range(len(vals))}
            # print(img_name,":",dict_rd[img_name])
            # print("======================")

        # Further checking and verification
        if "Average" in dict_rd:
            for key in list_header:
                vals = [dict_rd[img_name].get(key, None) for img_name in dict_rd if img_name != "Average"]
                avg = np.mean(vals, dtype=np.float64)
                assert abs(avg-dict_rd["Average"].get(key, None)) < 1e-2, "The mean of values is not equal to what is stored in Average"
                # print(key, "=", avg)
            if deleteAverageField:
                del dict_rd["Average"]  # Delete the Average

    # print(dict_rd)
    return dict_rd


def bits2Bytes(bits, to, bsize=1024):
    """convert bytes to megabytes, etc.
       sample code:
           print('mb= ' + str(bytesto(314575262000000, 'm')))
       sample output:
           mb= 300002347.946
    """

    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = [float(b)/8 for b in bits]
    for i in range(a[to]):
        r = [b / bsize for b in r]

    return(r)
