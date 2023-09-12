import torch
import projects.deep_video_compression.utils.pytorch as th_utils

class SpherePixelShuffle(torch.nn.Module):
    def __init__(self,
                 upscale_factor,
                 node_dim):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.node_dim = node_dim

    def forward(self, input):
        return th_utils.pixel_shuffle_1d(input, self.upscale_factor, self.node_dim)