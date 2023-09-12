import torch as th
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import random

def custom_max (x, dim):
    return th.max (x, dim=dim).values


def hp_pool (parents, x, fn):
    idx = 0

    x_k = th.empty (len(parents.keys()), x.size(1), dtype=x.dtype)

    for k, v in parents.items ():
        x_k[idx] = fn(x[v], dim=0)
        idx = idx + 1

    return x_k


def hp_max_pool (parents, x):
    return hp_pool(parents, x, custom_max)


def hp_avg_pool (parents, x):
    return hp_pool(parents, x, th.mean)


def hp_unpool (parents, x, newsize):
    idx = 0

    x_k = th.empty (newsize, x.shape[1], dtype=x.dtype)

    for k, v in parents.items ():
        x_k[v] = x[idx]
        idx = idx + 1

    return x_k

def pixel_shuffle_1d(input, upscale_factor, node_dim):
    "node_dim is the dimension related to node dim. It should be either 1 or 2, because dims[0] always refers to batche_size"
    dims = input.size()
    assert len(dims) == 3, "Only support 3d input"
    upscale_square = upscale_factor * upscale_factor

    if node_dim == 1:
        # in healpix the images_equirectangular are organized as [batch_size, num_nodes, num_features]
        output_channel = dims[2] // upscale_square
        input_view = input.reshape(dims[0], dims[1], output_channel, upscale_square)
        return input_view.permute(0, 1, 3, 2).reshape(dims[0], dims[1] * upscale_square, output_channel)
    else:   # node_dim = 2
        output_channel = dims[1] // upscale_factor
        input_view = input.reshape(dims[0], output_channel, upscale_factor, dims[1])
        return input_view.permute(0, 1, 3, 2).reshape(dims[0], output_channel, dims[1] * upscale_factor)


def plot_grad_flow(named_parameters, filename):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())


    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Value")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-absolute-gradient', 'mean-absolute-gradient', 'zero-gradient'])
    if filename:
        np.savez(filename+".npz", max_grads=max_grads, ave_grads=ave_grads)
        plt.savefig(filename+".png", bbox_inches='tight', dpi=300)
        plt.close()
        # npzfile = np.load(filename+".npz")
        # test_max_grads = npzfile['max_grads']
        # test_ave_grads = npzfile['ave_grads']
    else:
        plt.show()

# Function for setting the seed
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available(): # GPU operation have separate seed
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    th.backends.cudnn.determinstic = True
    th.backends.cudnn.benchmark = False