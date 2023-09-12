
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ops.parametrizers import NonNegativeParametrizer


class SphereGDN(nn.Module):
    r"""Generalized Divisive Normalization layer for Spherical images_equirectangular.

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """
    def __init__(self,
                 in_channels,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x):

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        norm = F.linear(x ** 2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out