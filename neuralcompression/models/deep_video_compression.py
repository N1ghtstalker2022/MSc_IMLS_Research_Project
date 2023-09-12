# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Code is based on the papers:

Lu, Guo, et al. "DVC: An end-to-end deep video compression framework."
CVPR (2019).

Yang, Ren et al.
"OpenDVC: An Open Source Implementation of the DVC Video Compression Method"
arXiv:2006.15862
"""
import sys
import time

import cv2
import math
import os
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from compressai.entropy_models import EntropyBottleneck
from matplotlib import pyplot as plt
from torch import Tensor, nn

import neuralcompression.functional as ncF
from neuralcompression.layers import SimplifiedGDN, SimplifiedInverseGDN
import healpy as hp
from compressai.ops.parametrizers import NonNegativeParametrizer

sys.path.append("../..")
from projects.deep_video_compression.utils import healpix as hp_utils


# import projects.deep_video_compression.utils.pytorch as th_utils


class CompressedPFrame(NamedTuple):
    """
    Output of DVC model compressing a single frame.

    Args:
        compressed_flow: Flow fields compressed to a string.
        flow_decomp_sizes: Metadata for the size of the flow representations at
            each compression level.
        compressed_residual: Residual compressed to a string.
        residual_decomp_sizes: Metadata for the size of the residual
            representations at each compression level.
    """

    compressed_flow: Any
    flow_decomp_sizes: List[torch.Size]
    compressed_residual: Any
    residual_decomp_sizes: List[torch.Size]


class DVCTrainOutput(NamedTuple):
    """
    Output of DVC forward function.

    Args:
        flow: The calculated optical flow field.
        image2_est: An estimate of image2 using the flow field and residual.
        residual: The residual between the flow-compensated ``image1`` and the
            true ``image2``.
        flow_probabilities: Estimates of probabilities of each compressed flow
            latent using entropy bottleneck.
        resid_probabilities: Estimates of probabilities of each compressed
            residual latent using entropy bottleneck.
    """

    flow: Tensor
    image2_est: Tensor
    residual: Tensor
    flow_probabilities: Tensor
    resid_probabilities: Tensor


class DVC(nn.Module):
    """
    Deep Video Compression composition class.

    This composes the Deep Video Compression Module of Lu (2019). It requires
    defining a motion estimator, a motion autoencoder, a motion compensator,
    and a residual autoencoder. The individual modules can be input by the
    user. If the user does not input a module, then this class will construct
    the module with default parameters from the paper.

    Lu, Guo, et al. "DVC: An end-to-end deep video compression framework."
    CVPR (2019).

    Args:
        coder_channels: Number of channels to use for autoencoders.
        motion_estimator: A module for estimating motion. See
            ``DVCPyramidFlowEstimator`` for an example.
        motion_encoder: A module for encoding motion fields. See
            ``DVCCompressionEncoder`` for an example.
        motion_entropy_bottleneck: A module for quantization and latent
            probability estimation for the compressed motion fields. See
            ``EntropyBottleneck`` for an example.
        motion_decoder: A module for decoding motion fields. See
            ``DVCCompressionDecoder`` for an example.
        motion_compensation: A module for compensating for motion errors. See
            ``DVCMotionCompensationModel`` for an example.
        residual_encoder: A module for encoding residuals after motion
            compensation. See ``DVCCompressionEncoder`` with 3 input channels
            for an example.
        residual_entropy_bottleneck: A module for quantization and latent
            probability estimation for the compressed residuals. See
            ``EntropyBottleneck`` for an example.
        residual_decoder: A module for decoding residuals. See
            ``DVCCompressionEncoder`` with 3 output channels for an example.
    """

    def __init__(
            self,
            coder_channels: int = 128,
            motion_estimator: Optional[nn.Module] = None,
            motion_encoder: Optional[nn.Module] = None,
            motion_entropy_bottleneck: Optional[nn.Module] = None,
            motion_decoder: Optional[nn.Module] = None,
            motion_compensation: Optional[nn.Module] = None,
            residual_encoder: Optional[nn.Module] = None,
            residual_entropy_bottleneck: Optional[nn.Module] = None,
            residual_decoder: Optional[nn.Module] = None,
            residual_network: Optional[nn.Module] = None
    ):
        super().__init__()

        self.motion_estimator = (
            DVCPyramidFlowEstimator() if motion_estimator is None else motion_estimator
        )
        self.motion_encoder = (
            DVCCompressionEncoder(
                filter_channels=coder_channels, out_channels=coder_channels
            )
            if motion_encoder is None
            else motion_encoder
        )
        self.motion_entropy_bottleneck = (
            EntropyBottleneck(coder_channels)
            if motion_entropy_bottleneck is None
            else motion_entropy_bottleneck
        )
        self.motion_decoder = (
            DVCCompressionDecoder(
                in_channels=coder_channels, filter_channels=coder_channels
            )
            if motion_decoder is None
            else motion_decoder
        )
        self.motion_compensation = (
            DVCMotionCompensationModel()
            if motion_compensation is None
            else motion_compensation
        )
        self.residual_encoder = (
            DVCResidualEncoder(
                in_channels=3,
                # filter_channels=coder_channels,
                out_channels=coder_channels,
                kernel_size=5,
            )
            if residual_encoder is None
            else residual_encoder
        )
        self.residual_entropy_bottleneck = (
            EntropyBottleneck(coder_channels)
            if residual_entropy_bottleneck is None
            else residual_entropy_bottleneck
        )
        self.residual_decoder = (
            DVCResidualDecoder(
                in_channels=coder_channels,
                # filter_channels=coder_channels,
                out_channels=3,
                kernel_size=5,
            )
            if residual_decoder is None
            else residual_decoder
        )
        self.residual_network = residual_network
        self.motion_entropy_bottleneck.update(force=True)
        self.residual_entropy_bottleneck.update(force=True)

    def compress(self, image1: Tensor, image2: Tensor) -> CompressedPFrame:
        """
        Compute compressed motion and residual between two images.

        Note:
            You must call ``update`` prior to calling ``compress``.

        Args:
            image1: The first image.
            image2: The second image.

        Returns:
            CompressedPFrame tuple with 1) the compressed flow field, 2) the
            flow decomposition sizes, 3) the compressed residual, and 4) the
            residual decomposition sizes.
        """
        flow = self.motion_estimator(image1, image2)

        # compress optical flow fields
        flow_latent, flow_decomp_sizes = self.motion_encoder(flow)
        compressed_flow = self.motion_entropy_bottleneck.compress(flow_latent)
        flow_latent = self.motion_entropy_bottleneck.decompress(
            compressed_flow, flow_latent.shape[-2:]
        )
        flow = self.motion_decoder(flow_latent, flow_decomp_sizes)

        # apply optical flow fields
        image2_est = ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1))

        # compensate for optical flow errors
        image2_est = self.motion_compensation(image1, image2_est, flow)

        # encode final residual
        residual = image2 - image2_est
        residual_latent, resid_decomp_sizes = self.residual_encoder(residual)
        compressed_residual = self.residual_entropy_bottleneck.compress(residual_latent)

        return CompressedPFrame(
            compressed_flow,
            flow_decomp_sizes,
            compressed_residual,
            resid_decomp_sizes,
        )

    def decompress(self, image1: Tensor, compressed_pframe: CompressedPFrame) -> Tensor:
        """
        Decompress motion fields and residual and compute next frame estimate.

        Args:
            image1: The base image for computing the next frame estimate.
            compressed_pframe: A compressed P-frame and metadata.

        Returns:
            An estimate of ``image2`` using ``image1`` and the compressed
                transition information.
        """
        flow_latent_size = (
            compressed_pframe.flow_decomp_sizes[-1][-2] // 2,
            compressed_pframe.flow_decomp_sizes[-1][-1] // 2,
        )
        flow_latent = self.motion_entropy_bottleneck.decompress(
            compressed_pframe.compressed_flow, flow_latent_size
        )
        flow = self.motion_decoder(flow_latent, compressed_pframe.flow_decomp_sizes)

        # apply optical flow fields
        image2_est = ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1))

        # compensate for optical flow errors
        image2_est = self.motion_compensation(image1, image2_est, flow)

        # decode residual
        residual_latent_size = (
            compressed_pframe.residual_decomp_sizes[-1][-2] // 2,
            compressed_pframe.residual_decomp_sizes[-1][-1] // 2,
        )
        residual_latent = self.residual_entropy_bottleneck.decompress(
            compressed_pframe.compressed_residual, residual_latent_size
        )
        residual = self.residual_decoder(
            residual_latent, compressed_pframe.residual_decomp_sizes
        )

        return (image2_est + residual).clamp_(0, 1)

    def update(self, force: bool = True) -> bool:
        """
        Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force: Overwrite previous values.
        """
        update1 = self.motion_entropy_bottleneck.update(force=force)
        update2 = self.residual_entropy_bottleneck.update(force=force)

        return update1 or update2

    def forward(self, image1: Tensor, image2: Tensor, dict_index=None, dict_weight=None, sample_res=None,
                patch_res=None, patch_id=None, nPix_per_patch=None) -> DVCTrainOutput:
        """
        Apply DVC coding to a pair of images.

        The ``forward`` function is expected to be used for training. For
        inference, see the ``compress`` and ``decompress``.

        Args:
            image1: The base image.
            image2: The second image. Optical flow will be applied to
                ``image1`` to predict ``image2``.

        Returns:
            A 5-tuple training output containing 1) the calculated optical
            flow, 2) an estimate of ``image2`` including motion compensation,
            3) the residual between ``image2`` and its estimate, 4) the
            probabilities output by the flow quantizer, and 5) the
            probabilities output by the residual quantizer.
        """
        # estimate optical flow fields
        flow = self.motion_estimator(image1, image2)
        ori_flow = flow
        # compress optical flow fields

        # TODO: the flow, it does not learn anything?
        flow, sizes = self.motion_encoder(flow)
        flow, flow_probabilities = self.motion_entropy_bottleneck(flow)
        flow = self.motion_decoder(flow, sizes)

        # plot flow
        # Compute the figure size
        # fig_height_per_subplot = 256 / 80  # assuming 80 dpi, you can adjust this
        # fig_width_per_subplot = 448 / 80  # assuming 80 dpi, you can adjust this
        # fig_width = 2 * fig_width_per_subplot
        # fig_height = fig_height_per_subplot
        #
        # # Create a new figure
        # fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        #
        # # third_channel = 0 * np.ones((256, 448))
        # ori_flow_plot = ori_flow.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy()
        # # ori_flow_plot = np.concatenate((ori_flow_plot, third_channel[:, :, np.newaxis]), axis=2)
        #
        # flow_plot = flow.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy()
        # # flow_plot = np.concatenate((flow_plot, third_channel[:, :, np.newaxis]), axis=2)
        #
        #
        # # Plot second sequence of images in the second row
        # axes[0].imshow(ori_flow_plot[:, :, 0])
        # axes[0].set_title(f"Ori_Flow")
        # axes[0].axis('off')
        #
        # axes[1].imshow(flow_plot[:, :, 0])
        # axes[1].set_title(f"Flow")
        # axes[1].axis('off')
        #
        # prefix = 'performance'
        # path = prefix + '/compare-flow' + '.png'
        # if not os.path.exists(prefix):
        #     os.makedirs(prefix)
        # plt.savefig(path)
        # plt.clf()

        # apply optical flow fields
        image2_est = ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1))
        warped_image2_est = image2_est.clone()

        # compensate for optical flow errors
        image2_est = self.motion_compensation(image1, image2_est, flow)
        image2_compensated = image2_est.clone()
        # encode final residual
        residual = image2 - image2_est  # residual size: [batch_size=4, 3, 128, 256]
        ori_residual = residual.clone()
        img_size = residual.shape[-2:]

        prefix = 'performance'
        path = prefix + '/compare-flow' + '.png'
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        # warped_image2_est_save = (warped_image2_est.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy() * 255).astype(np.uint8)
        # cv2.imwrite("performance/warped-frame.png", cv2.cvtColor(warped_image2_est_save, cv2.COLOR_RGB2BGR))
        # image2_compensated_save = (image2_compensated.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy() * 255).astype(np.uint8)
        # cv2.imwrite("performance/compensated-frame.png", cv2.cvtColor(image2_compensated_save, cv2.COLOR_RGB2BGR))
        # image2_save = (image2.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy() * 255).astype(np.uint8)
        # cv2.imwrite("performance/current-frame.png", cv2.cvtColor(image2_save, cv2.COLOR_RGB2BGR))
        # ori_residual_save = (ori_residual.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy() * 255).astype(np.uint8)
        # cv2.imwrite("performance/residual.png", cv2.cvtColor(ori_residual_save, cv2.COLOR_RGB2BGR))

        if self.residual_network is not None:
            healpix_res = sample_res  # sample_res = 7
            nside = hp.order2nside(healpix_res)

            residual = residual.permute(0, 2, 3, 1)  # residual size: [batch_size=4, 128, 256, 3]
            residual_on_sphere = hp_utils.sampleEquirectangularForHEALPix(residual, nside, interpolation='bilinear',
                                                                          nest=True)
            # residual_on_sphere size: [batch_size=4, 49152, 3]
            residual_on_sphere_ori = residual_on_sphere
            # residual_on_sphere_ori_size = residual_on_sphere.shape

            # Get a patch of residual_on_sphere to train. Save computing resource.
            if patch_res is not None:
                residual_on_sphere = residual_on_sphere.narrow(dim=1, start=patch_id * nPix_per_patch,
                                                               length=nPix_per_patch)  # start=43*1024, length=1024

                residual_dict = self.residual_network(residual_on_sphere, dict_index, dict_weight, sample_res, patch_res)
                resid_probabilities = residual_dict['likelihoods']['y']
                c = resid_probabilities.detach().cpu().numpy()
                residual_on_sphere = residual_dict['x_hat']
                residual_on_sphere_ori[:, patch_id * nPix_per_patch:patch_id * nPix_per_patch + nPix_per_patch,
                :] = residual_on_sphere

                residual_on_sphere = residual_on_sphere_ori
            else:
                residual_dict = self.residual_network(residual_on_sphere, dict_index, dict_weight, sample_res, patch_res)
                resid_probabilities = residual_dict['likelihoods']['y']
                residual_on_sphere = residual_dict['x_hat']
            # TODO: hard-coded for now
            recovered_residual = healpix_to_erp(residual_on_sphere, img_size[0], img_size[1])
            residual = recovered_residual.permute(0, 3, 1, 2)
            resid_probabilities = resid_probabilities.permute(0, 2, 1)
            # add some plot

        else:
            residual, sizes = self.residual_encoder(residual)  # residual size: [batch_size=4, 128, 8, 16]
            # new residual size = [4, 128, 49142]

            residual, resid_probabilities = self.residual_entropy_bottleneck(
                residual)  # residual size: [batch_size=4, 128, 8, 16]

            residual = self.residual_decoder(residual, sizes)  # residual size: [batch_size=4, 3, 128, 256]
        image2_est = image2_est + residual

        # # Create a new figure
        # fig, axes = plt.subplots(2, 2)
        #
        #
        # axes[0, 0].imshow(image2.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
        # axes[0, 0].text(0.5, -0.2, 'Current Frame', ha='center', va='center', transform=axes[0, 0].transAxes)
        # axes[0, 0].axis('off')
        # axes[0, 1].imshow(image2_est.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
        # axes[0, 1].text(0.5, -0.2, 'Compensated Frame', ha='center', va='center', transform=axes[0, 1].transAxes)
        # axes[0, 1].axis('off')
        #
        # axes[1, 0].imshow(ori_residual.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
        # axes[1, 0].text(0.5, -0.2, 'Optical Flow', ha='center', va='center', transform=axes[1, 0].transAxes)
        # axes[1, 0].axis('off')
        #
        # axes[1, 1].imshow(residual.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
        # axes[1, 1].text(0.5, -0.2, 'Reconstructed Optical Flow', ha='center', va='center', transform=axes[1, 1].transAxes)
        # axes[1, 1].axis('off')
        #
        # plt.savefig(path)
        # plt.clf()
        #
        # plt.imshow(warped_image2_est.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
        # plt.axis('off')  # Turn off the axis
        # plt.savefig('warped-frame.png', bbox_inches='tight', pad_inches=0)
        # plt.clf()
        #
        # plt.imshow(image2_compensated.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
        # plt.axis('off')  # Turn off the axis
        # plt.savefig('compensated-frame.png', bbox_inches='tight', pad_inches=0)
        # plt.clf()
        #
        # plt.imshow(image2.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
        # plt.axis('off')  # Turn off the axis
        # plt.savefig('current-frame.png', bbox_inches='tight', pad_inches=0)
        # plt.clf()
        #
        plt.imshow(image2_est.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
        plt.axis('off')  # Turn off the axis
        plt.savefig('reconstructed-frame.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        #
        # plt.imshow(ori_residual.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
        # plt.axis('off')  # Turn off the axis
        # plt.savefig('residual.png', bbox_inches='tight', pad_inches=0)
        # plt.clf()
        #
        # plt.imshow(residual.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
        # plt.axis('off')  # Turn off the axis
        # plt.savefig('reconstructed-residual.png', bbox_inches='tight', pad_inches=0)
        # plt.clf()


        # image2_est_save = (image2_est.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy() * 255).astype(np.uint8)
        # cv2.imwrite("performance/reconstructed-frame.png", cv2.cvtColor(image2_est_save, cv2.COLOR_RGB2BGR))
        # reconstructed_residual_save = (residual.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy() * 255).astype(np.uint8)
        # cv2.imwrite("performance/reconstructed-residual.png", cv2.cvtColor(reconstructed_residual_save, cv2.COLOR_RGB2BGR))


        # psnr_metric = torchmetrics.PSNR(data_range=1.0)
        # print(psnr_metric(image2_est, image2))
        # print(psnr_metric(image2_compensated, image2))


        return DVCTrainOutput(
            flow, image2_est, residual, flow_probabilities, resid_probabilities
        )


class DVCCompressionDecoder(nn.Module):
    """
    Deep Video Compression decoder module.

    Args:
        in_channels: Number of channels in latent space.
        out_channels: Number of channels to output from decoder.
        filter_channels: Number of channels for intermediate layers.
        kernel_size: Size of convolution kernels.
        stride: Stride of convolutions.
        num_conv_layers: Number of convolution layers.
        use_gdn: Whether to use Generalized Divisive Normalization or
            BatchNorm.
    """

    def __init__(
            self,
            in_channels: int = 128,
            out_channels: int = 2,
            filter_channels: int = 128,
            kernel_size: int = 3,
            stride: int = 2,
            num_conv_layers: int = 4,
            use_gdn: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        bias = True if use_gdn else False

        self.layers = nn.ModuleList()
        for _ in range(num_conv_layers - 1):
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=filter_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
            if use_gdn:
                self.layers.append(SimplifiedInverseGDN(filter_channels))
            else:
                self.layers.append(nn.BatchNorm2d(filter_channels))
                self.layers.append(nn.ReLU())

            in_channels = filter_channels

        self.layers.append(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=True,
            )
        )

    def forward(
            self, image: Tensor, output_sizes: Optional[Sequence[torch.Size]] = None
    ) -> Tensor:
        # validate output_sizes
        if output_sizes is not None:
            output_sizes = [os for os in output_sizes]  # shallow copy
            transpose_conv_count = 0
            for layer in self.layers:
                if isinstance(layer, nn.ConvTranspose2d):
                    transpose_conv_count += 1
            if not transpose_conv_count == len(output_sizes):
                raise ValueError(
                    "len(output_sizes) must match number of transpose convolutions."
                )

        # run the deconvolutions
        for layer in self.layers:
            # use sizes from encoder if we have them for decoding
            if isinstance(layer, nn.ConvTranspose2d) and output_sizes is not None:
                image = layer(image, output_sizes.pop())
            else:
                image = layer(image)

        return image


# class SDPAConv(torch.nn.Module):
#     r"""Class for implementing Sphere Directional and Position-Aware convolution
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size, node_dim=1, bias=True):
#         super(SDPAConv, self).__init__()
#
#         assert node_dim >= 0
#         self.node_dim = node_dim
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#
#         self.weight = torch.nn.Parameter(torch.Tensor(kernel_size, in_channels, out_channels))
#
#         if bias:
#             self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         # Took it from torch.nn.Conv2d()
#
#         torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         # torch.nn.init.xavier_uniform_(self.weight, gain=2.)
#         if self.bias is not None:
#             fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             torch.nn.init.uniform_(self.bias, -bound, bound)
#         # Took it from torch_geometric.nn.ChebConv
#         # torch_geometric.nn.inits.glorot(self.weight)
#         # torch_geometric.nn.inits.zeros(self.bias)
#
#     def forward(self, x, neighbors_indices, neighbors_weights, valid_index=None):
#         assert (self.kernel_size - 1) == neighbors_weights.size(1), "size does not match"
#
#         out = torch.matmul(x, self.weight[0])
#
#         # We can precompute some of these repeated operations outside the loop:
#         if valid_index is None:
#             precomputed_results = [torch.mul(neighbors_weights.narrow(dim=1, start=k, length=1),
#                                              x.index_select(self.node_dim, neighbors_indices[:, k]))
#                                    for k in range(self.kernel_size - 1)]
#
#             for k, precomp in enumerate(precomputed_results, start=1):
#                 out += torch.matmul(precomp, self.weight[k])
#         else:
#             for k in range(1, self.kernel_size):
#                 col = k - 1
#                 valid_rows = valid_index[:, col]
#                 s = torch.mul(neighbors_weights[valid_rows, col].view(-1, 1),
#                               x.index_select(self.node_dim, neighbors_indices[valid_rows, col]))
#                 out[:, valid_rows, :] += torch.matmul(s, self.weight[k])
#
#         if self.bias is not None:
#             out += self.bias
#
#         return out


# class DVCResidualEncoder(nn.Module):
#     # for spherical convolutions
#     def __init__(
#             self,
#             neighbors_indices,
#             neighbors_weights,
#             in_channels: int = 3,
#             out_channels: int = 128,
#             filter_channels: int = 128,
#             kernel_size: int = 9,
#             # stride: int = 2,
#             num_conv_layers: int = 4,
#             use_gdn: bool = True,
#
#     ):
#         super().__init__()
#         bias = True if use_gdn else False
#         self.activation = SphereGDN(filter_channels, False)
#         self.pool = getattr(torch.nn, "AvgPool3d")(kernel_size=(1, 4, 1))
#
#         self.layers = nn.ModuleList()
#         for _ in range(num_conv_layers - 1):
#             # no stride options for current spherical convolutions (i.e. stride=1)
#             self.layers.append(
#                 SDPAConv(
#                     in_channels=in_channels,
#                     out_channels=filter_channels,
#                     kernel_size=kernel_size,
#                     bias=bias,
#                 )
#             )
#             self.layers.append(self.activation)
#             self.layers.append(self.pool)
#             # average pooling for spherical convolutions
#
#             # if use_gdn:
#             #     self.layers.append(SimplifiedGDN(filter_channels))
#             # else:
#             #     self.layers.append(nn.BatchNorm2d(filter_channels))
#             #     self.layers.append(nn.ReLU())
#
#             in_channels = filter_channels
#
#         self.layers.append(
#             SDPAConv(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 bias=True,
#             )
#         )
#
#         self.neighbors_indices = neighbors_indices
#         self.neighbors_weights = neighbors_weights
#
#     def forward(self, image: Tensor) -> Tuple[Tensor, List[torch.Size]]:
#
#         # TODO: 1: apply healpix transform to residual
#         #  -> 2:test it
#         healpix_res = 5
#         nside = hp.order2nside(healpix_res)
#
#         residual = image.permute(0, 2, 3, 1)  # residual size: [batch_size=4, 128, 256, 3]
#         image_on_sphere = sampleEquirectangularForHEALPix(residual, nside, interpolation='bilinear')
#
#         # determine neighbors for each pixel on sphere.
#
#         # image_on_sphere is a tensor of size [batch_size, num_nodes, num_features]
#         sizes = []
#         for layer in self.layers:
#             # log the sizes so that we can recover them with strides != 1
#             if isinstance(layer, SDPAConv):
#                 sizes.append(image_on_sphere.shape)
#             if layer.__class__.__name__ == "SDPAConv":
#                 image_on_sphere = layer(image_on_sphere, self.neighbors_indices, self.neighbors_weights)
#             # non-linear activation layer
#             elif layer.__class__.__name__ == "SphereGDN":
#
#                 image_on_sphere = layer(image_on_sphere)
#             # pooling layer
#             else:
#                 image_on_sphere = torch.squeeze(self.pool(torch.unsqueeze(image_on_sphere, dim=0)), dim=0)
#         # image_on_sphere = self.activation(image_on_sphere)
#         image_on_sphere = image_on_sphere.permute(0, 2, 1)  # residual size: [batch_size=4, 128, 49142]
#         return image_on_sphere, sizes

# class DVCResidualDecoder(nn.Module):
#     # for spherical convolutions
#     def __init__(
#             self,
#             neighbors_indices,
#             neighbors_weights,
#             in_channels: int = 128,
#             out_channels: int = 3,
#             filter_channels: int = 128,
#             kernel_size: int = 9,
#             # stride: int = 2,
#             num_conv_layers: int = 4,
#             use_gdn: bool = True,
#     ):
#         super().__init__()
#         bias = True if use_gdn else False
#         self.activation = SphereGDN(filter_channels, True)
#         self.unpool = SpherePixelShuffle(2, 1)
#
#         self.layers = nn.ModuleList()
#         for _ in range(num_conv_layers - 1):
#
#             # no stride options for current spherical convolutions (i.e. stride=1)
#             self.layers.append(
#                 SDPAConv(
#                     in_channels=in_channels,
#                     out_channels=filter_channels * 4,  # 4 is the number of pixels in the unpooling kernel
#                     kernel_size=kernel_size,
#                     bias=bias,
#                 )
#             )
#             self.layers.append(self.unpool)
#
#             self.layers.append(self.activation)
#
#
#
#             # if use_gdn:
#             #     self.layers.append(SimplifiedGDN(filter_channels))
#             # else:
#             #     self.layers.append(nn.BatchNorm2d(filter_channels))
#             #     self.layers.append(nn.ReLU())
#
#             in_channels = filter_channels
#
#         self.layers.append(
#             SDPAConv(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 bias=True,
#             )
#         )
#
#         # self.activation = SphereGDN(out_channels, True)
#         self.neighbors_indices = neighbors_indices
#         self.neighbors_weights = neighbors_weights
#
#
#
#     def forward(self, image_on_sphere: Tensor, output_sizes: Optional[Sequence[torch.Size]] = None) -> Tensor:
#         image_on_sphere = image_on_sphere.permute(0, 2, 1) # [4, 49152, 3]
#
#         # determine neighbors for each pixel on sphere.
#         # validate output_sizes
#         if output_sizes is not None:
#             output_sizes = [os for os in output_sizes]  # shallow copy
#             transpose_conv_count = 0
#             for layer in self.layers:
#                 if isinstance(layer, SDPAConv):
#                     transpose_conv_count += 1
#             if not transpose_conv_count == len(output_sizes):
#                 raise ValueError(
#                     "len(output_sizes) must match number of transpose convolutions."
#                 )
#
#         #
#
#         for layer in self.layers:
#             # log the sizes so that we can recover them with strides != 1
#             # if isinstance(layer, SDPAConv) and output_sizes is not None:
#             #     image_on_sphere = layer(image_on_sphere, self.neighbors_indices, self.neighbors_weights)
#             # else:
#             if layer.__class__.__name__ == "SDPAConv":
#                 image_on_sphere = layer(image_on_sphere, self.neighbors_indices, self.neighbors_weights)
#             else:
#                 # pooling layer or non-linear activation layer
#                 image_on_sphere = layer(image_on_sphere)
#
#         # TODO: hard-coded for now
#         recovered_residual = healpix_to_erp(image_on_sphere, 64, 128)
#
#         recovered_residual = recovered_residual.permute(0, 3, 1, 2)
#         return recovered_residual


class DVCCompressionEncoder(nn.Module):
    """
    Deep Video Compression encoder module.

    Args:
        in_channels: Number of channels in image space.
        out_channels: Number of channels to output from encoder.
        filter_channels: Number of channels for intermediate layers.
        kernel_size: Size of convolution kernels.
        stride: Stride of convolutions.
        num_conv_layers: Number of convolution layers.
        use_gdn: Whether to use Generalized Divisive Normalization or
            BatchNorm.
    """

    def __init__(
            self,
            in_channels: int = 2,
            out_channels: int = 128,
            filter_channels: int = 128,
            kernel_size: int = 3,
            stride: int = 2,
            num_conv_layers: int = 4,
            use_gdn: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        bias = True if use_gdn else False

        self.layers = nn.ModuleList()
        for _ in range(num_conv_layers - 1):
            self.layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filter_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
            if use_gdn:
                self.layers.append(SimplifiedGDN(filter_channels))
            else:
                self.layers.append(nn.BatchNorm2d(filter_channels))
                self.layers.append(nn.ReLU())

            in_channels = filter_channels

        self.layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=True,
            )
        )

    def forward(self, image: Tensor) -> Tuple[Tensor, List[torch.Size]]:
        sizes = []
        for layer in self.layers:
            # log the sizes so that we can recover them with strides != 1
            if isinstance(layer, nn.Conv2d):
                sizes.append(image.shape)

            image = layer(image)

        return image, sizes


class DVCResidualEncoder(nn.Module):
    """
    Deep Video Compression encoder module.

    Args:
        in_channels: Number of channels in image space.
        out_channels: Number of channels to output from encoder.
        filter_channels: Number of channels for intermediate layers.
        kernel_size: Size of convolution kernels.
        stride: Stride of convolutions.
        num_conv_layers: Number of convolution layers.
        use_gdn: Whether to use Generalized Divisive Normalization or
            BatchNorm.
    """

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 128,
            filter_channels: int = 64,
            kernel_size: int = 3,
            stride: int = 2,
            num_conv_layers: int = 4,
            use_gdn: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        bias = True if use_gdn else False
        N = filter_channels
        M = out_channels
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=N,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            )
        )
        self.layers.append(
            nn.Conv2d(
                in_channels=N,
                out_channels=N,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            )
        )
        if use_gdn:
            self.layers.append(SimplifiedGDN(N))
        else:
            self.layers.append(nn.BatchNorm2d(N))
            self.layers.append(nn.ReLU())

        self.layers.append(
            nn.Conv2d(
                in_channels=N,
                out_channels=M,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            )
        )
        self.layers.append(
            nn.Conv2d(
                in_channels=M,
                out_channels=M,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            )
        )

    def forward(self, image: Tensor) -> Tuple[Tensor, List[torch.Size]]:
        sizes = []
        for layer in self.layers:
            # log the sizes so that we can recover them with strides != 1
            if isinstance(layer, nn.Conv2d):
                sizes.append(image.shape)

            image = layer(image)

        return image, sizes


class DVCResidualDecoder(nn.Module):
    """
    Deep Video Compression decoder module.

    Args:
        in_channels: Number of channels in latent space.
        out_channels: Number of channels to output from decoder.
        filter_channels: Number of channels for intermediate layers.
        kernel_size: Size of convolution kernels.
        stride: Stride of convolutions.
        num_conv_layers: Number of convolution layers.
        use_gdn: Whether to use Generalized Divisive Normalization or
            BatchNorm.
    """

    def __init__(
            self,
            in_channels: int = 128,
            out_channels: int = 3,
            filter_channels: int = 64,
            kernel_size: int = 3,
            stride: int = 2,
            num_conv_layers: int = 2,
            use_gdn: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        bias = True if use_gdn else False
        # TODO: change this!
        N = filter_channels
        M = in_channels
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ConvTranspose2d(
                in_channels=M,
                out_channels=N,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            )
        )
        self.layers.append(
            nn.ConvTranspose2d(
                in_channels=N,
                out_channels=N,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            )
        )
        if use_gdn:
            self.layers.append(SimplifiedInverseGDN(filter_channels))
        else:
            self.layers.append(nn.BatchNorm2d(filter_channels))
            self.layers.append(nn.ReLU())


        self.layers.append(
            nn.ConvTranspose2d(
                in_channels=N,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            )
        )
        self.layers.append(
            nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            )
        )



    def forward(
            self, image: Tensor, output_sizes: Optional[Sequence[torch.Size]] = None
    ) -> Tensor:
        # validate output_sizes
        if output_sizes is not None:
            output_sizes = [os for os in output_sizes]  # shallow copy
            transpose_conv_count = 0
            for layer in self.layers:
                if isinstance(layer, nn.ConvTranspose2d):
                    transpose_conv_count += 1
            if not transpose_conv_count == len(output_sizes):
                raise ValueError(
                    "len(output_sizes) must match number of transpose convolutions."
                )

        # run the deconvolutions
        for layer in self.layers:
            # use sizes from encoder if we have them for decoding
            if isinstance(layer, nn.ConvTranspose2d) and output_sizes is not None:
                image = layer(image, output_sizes.pop())
            else:
                image = layer(image)

        return image


class DVCPyramidFlowEstimator(nn.Module):
    """
    Pyramidal optical flow estimation.

    This estimates the optical flow at `levels` different pyramidal scales. It
    should be trained in conjunction with a pyramidal optical flow supervision
    signal.

    Args:
        in_channels: Number of input channels. Typically uses two 3-channel
            images, plus an initial flow with 2 more channels.
        filter_counts: Number of filters for each stage of pyramid. Defaults
            to ``[32, 64, 32, 16, 2]``.
        kernel_size: Kernel size of filters.
        levels: Number of pyramid levels.
    """

    def __init__(
            self,
            in_channels: int = 8,
            filter_counts: Optional[Sequence[int]] = None,
            kernel_size: int = 7,
            levels: int = 5,
    ):
        super().__init__()
        if filter_counts is None:
            filter_counts = [32, 64, 32, 16, 2]
        padding = kernel_size // 2
        self.model_levels = nn.ModuleList()

        for _ in range(levels):
            current_in_channels = in_channels
            layers: List[nn.Module] = []
            for i, filter_count in enumerate(filter_counts):
                layers.append(
                    nn.Conv2d(
                        in_channels=current_in_channels,
                        out_channels=filter_count,
                        kernel_size=kernel_size,
                        padding=padding,
                    )
                )
                if i < len(filter_counts) - 1:
                    layers.append(nn.ReLU())

                current_in_channels = filter_count

            self.model_levels.append(nn.Sequential(*layers))

    def _decompose_images_to_pyramids(
            self, image1: Tensor, image2: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Pyramid flow structure, average down for each pyramid level."""
        images_1 = [image1]
        images_2 = [image2]
        for _ in range(1, len(self.model_levels)):
            images_1.append(F.avg_pool2d(images_1[-1], 2))
            images_2.append(F.avg_pool2d(images_2[-1], 2))

        return images_1, images_2

    def calculate_flow_with_image_pairs(
            self, image1: Tensor, image2: Tensor
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        Calculate optical flow and return image pairs for each pyramid level.

        During training, we optimize the optical flow by trying to match the
        flow at each pyramid scale to its target. This function computes the
        images output by the flow at each level of the pyramid and returns them
        all in addition to the final flow.

        Args:
            image1: The based image - a flow map will be computed to transform
                ``image1`` into ``image2``.
            image2: The target image.

        Returns:
            A 2-tuple containing:
                The pyramidal flow estimate.
                Base/target image pairs from each level of the pyramid that can
                    be input into a distortion function.
        """
        if not image1.shape == image2.shape:
            raise ValueError("Image shapes must match.")

        images_1, images_2 = self._decompose_images_to_pyramids(image1, image2)
        flow = torch.zeros_like(images_1[-1])[:, :2]  # just the first two channels

        # estimate flows at all levels and return warped images
        optical_flow_pairs = []
        for model_level in self.model_levels:
            image1 = images_1.pop()
            image2 = images_2.pop()
            flow = F.interpolate(flow, image1.shape[2:], mode="bilinear")
            flow = flow + model_level(
                torch.cat(
                    (
                        ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1)),
                        image2,
                        flow,
                    ),
                    dim=1,
                )
            )
            optical_flow_pairs.append(
                (ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1)), image2)
            )

        # we have to return all of the warped images at different levels for
        # training
        return flow, optical_flow_pairs

    def forward(self, image1: Tensor, image2: Tensor) -> Tensor:
        """
        Calculate optical flow using pyramidal structure and a learned model.

        Args:
            image1: The based image - a flow map will be computed to transform
                ``image1`` into ``image2``.
            image2: The target image.

        Returns:
            Pyramidal optical flow estimate.
        """
        return self.calculate_flow_with_image_pairs(image1, image2)[0]


class DVCMotionCompensationModel(nn.Module):
    """
    Motion compensation model.

    After applying optical flow, there remain residual errors. This model
    corrects for the errors based on the input image, the transformed image,
    and the optical flow field.

    Args:
        in_channels: Number of input channels. The ``forward`` function takes
            as input two 3-channel images, plus an optical flow field, so by
            default this is 8 channels.
        model_channels: Number of channels for convolution kernels.
        out_channels: Number of channels for output motion-compensated image.
        kernel_size: Size of convolution kernels.
        num_levels: Number of convolution levels.
    """

    def __init__(
            self,
            in_channels: int = 8,
            model_channels: int = 64,
            out_channels: int = 3,
            kernel_size: int = 3,
            num_levels: int = 3,
    ):
        super().__init__()
        padding = kernel_size // 2

        # simple input layer
        input_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=model_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # build the autoencoder portion from the ground up
        child = None
        for _ in range(num_levels):
            child = _MotionCompensationLayer(
                child, in_channels=model_channels, out_channels=model_channels
            )
        unet = child

        # make sure we turn on bias for the output
        output_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=model_channels,
                out_channels=model_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=model_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
        )

        assert unet is not None
        self.model = nn.Sequential(input_layer, unet, output_layer)

    def forward(self, image1: Tensor, image2: Tensor, flow: Tensor) -> Tensor:
        return self.model(torch.cat((image1, image2, flow), dim=1))


class _MotionCompensationLayer(nn.Module):
    """
    Intermediate U-Net style layer for motion compensation.

    Args:
        child: The motion compensation module executes a U-Net style
            convolution structure with an input convolution followed by a child
            ``_MotionCompensationLayer`` followed by an output convolution. If
            the layer is at the bottom of the 'U', then a ``None`` is input for
            ``child``.
        in_channels: Number of input channels for convolutions.
        out_channels: Number of outpout channels for convolutions.
    """

    def __init__(
            self,
            child: Optional[nn.Module] = None,
            in_channels: int = 64,
            out_channels: int = 64,
    ):
        super().__init__()
        self.child = child
        self.input_block = _ResidualBlock(
            in_channels=in_channels, out_channels=out_channels
        )
        self.output_block = _ResidualBlock(
            in_channels=in_channels, out_channels=out_channels
        )

    def forward(self, image: Tensor) -> Tensor:
        upsample_size = tuple(image.shape[2:])

        # residual block includes skipped connection
        image = self.input_block(image)

        # if we have a child, downsample, run it, and then upsample back
        if self.child is not None:
            image = image + F.interpolate(
                self.child(F.avg_pool2d(image, kernel_size=2)),
                size=upsample_size,
                mode="nearest",
            )

        # residual block includes skipped connection
        return self.output_block(image)


class _ResidualBlock(nn.Module):
    """
    Simple pre-activated ResNet-style block.

    Structure: input -> ReLU -> Conv2d -> ReLU -> Conv2d + input

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride of convolutions.
    """

    def __init__(
            self,
            kernel_size: int = 3,
            in_channels: int = 64,
            out_channels: int = 64,
            stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
        )

    def forward(self, image: Tensor) -> Tensor:
        return image + self.block(image)


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


# class SpherePixelShuffle(torch.nn.Module):
#     def __init__(self,
#                  upscale_factor,
#                  node_dim):
#         super().__init__()
#         self.upscale_factor = upscale_factor
#         self.node_dim = node_dim
#
#     def forward(self, input):
#         return th_utils.pixel_shuffle_1d(input, self.upscale_factor, self.node_dim)

def sampleEquirectangularForHEALPix(equi_img, nside, interpolation, indexes=None, nest=True):
    if indexes is None:
        nPix = hp.nside2npix(nside)
        indexes = torch.arange(nPix)
    device = equi_img.device
    latitude, longitude = hp.pix2ang(nside=nside, nest=nest, ipix=indexes)
    latitude = latitude.to(torch.float32)
    longitude = longitude.to(torch.float32)
    latitude = latitude.to(device)
    longitude = longitude.to(device)
    batch_size = equi_img.size(0)
    # Pre-allocate a tensor with zeros
    # Shape: (batch_size, H, W, C)
    sampled_images = torch.zeros((batch_size, nPix, 3), dtype=torch.float32)

    # Fill in the tensor with the sampled images
    for i in range(batch_size):
        sampled = sampleEquirectangular(equi_img[i], latitude, longitude, flip=True, interpolation=interpolation)
        sampled_images[i] = sampled

    return sampled_images


def sampleEquirectangular(equi_img, colat, lon, flip, interpolation):
    """ Sampling Equirectangular using bilinear interpolation"""
    assert colat.shape == lon.shape, "Shapes must be similar"
    assert interpolation in ["bilinear", "lanczos"], "Interpolation not defined"
    # equi_img shape: [3, 128, 256]

    if flip:
        equi_img = torch.flip(equi_img, [1])  # Assuming HxWxC format
    C = equi_img.shape[2]
    pi_gpu = torch.tensor(3.1416, device=lon.device)
    y = (colat / pi_gpu) * equi_img.shape[0]
    lon[lon < 0.] += 2 * pi_gpu
    lon[lon > 2 * pi_gpu] -= 2 * pi_gpu
    x = (lon / (2 * pi_gpu)) * equi_img.shape[1]

    if interpolation == "bilinear":
        y = torch.clamp(y - 0.5, 0, equi_img.shape[0] - 1)
        x = x - 0.5
        x[x < 0] += equi_img.shape[1]

        i0 = y.long()
        j0 = x.long()
        i1 = i0 + 1
        j1 = (j0 + 1) % equi_img.shape[1]

        frac_x = y - i0.float()
        frac_y = x - j0.float()
        # TODO: fix this problem
        values = (1. - frac_x).unsqueeze(-1) * (1. - frac_y).unsqueeze(-1) * equi_img[i0, j0, :] + frac_x.unsqueeze(
            -1) * (1. - frac_y).unsqueeze(-1) * equi_img[i1, j0, :] + (1. - frac_x).unsqueeze(-1) * frac_y.unsqueeze(
            -1) * equi_img[i0, j1, :] + frac_x.unsqueeze(-1) * frac_y.unsqueeze(-1) * equi_img[i1, j1, :]
    else:
        raise ValueError("Unknown interpolation method.")
    ans = values.reshape((*colat.shape, C))
    return ans


def sumOfAP(a, d, n):  # sum of Arithmetic Progression (forced to return an integer by // in //2)
    return (n * (2 * a + (n - 1) * d)) // 2


def checkConsecutive(l):
    # Check if list contains consecutive numbers
    return sorted(l) == list(range(min(l), max(l) + 1))


def healpix_to_erp(healpix_data, H, W, flip=True):
    # Assuming healpix_data has shape (batch_size, n_pixels, channels)
    nside = hp.npix2nside(healpix_data.shape[1])

    # Compute theta and phi for each pixel in the ERP image
    theta, phi = np.mgrid[0:np.pi:H * 1j, 0:2 * np.pi:W * 1j]

    # Convert theta and phi to HEALPix indices
    healpix_indices = hp.ang2pix(nside, theta, phi, nest=True)

    # Process each batch separately
    batch_size = healpix_data.shape[0]
    erp_images = []

    for i in range(batch_size):
        # Sample the HEALPix data using the indices
        erp_image = healpix_data[i, healpix_indices]
        if flip:
            erp_image = torch.flip(erp_image, [1])
        erp_images.append(erp_image)

    return torch.stack(erp_images)


# def getHEALPixNeighbours():
#     struct_loader = HealpixSdpaStructLoader(weight_type='identity',
#                                             use_geodesic=True,
#                                             use_4connectivity=False,
#                                             normalization_method='non',
#                                             cutGraphForPatchOutside=True,
#                                             load_save_folder='/scratch/zczqyc4/healpix_structs')
#     index, weight, valid_neighbors = struct_loader.getStruct(sampling_res=5, num_hops=1,
#                                                              patch_res=None,
#                                                              patch_id=0)
#
#     return index, weight, valid_neighbors

# class HealpixSdpaStructLoader:
#     def __init__(self, weight_type, use_geodesic, use_4connectivity, normalization_method, cutGraphForPatchOutside,
#                  load_save_folder=None):
#
#         self.weight_type = weight_type
#         self.use_geodesic = use_geodesic
#         self.use_4connectivity = use_4connectivity
#         self.isNest = True
#         self.folder = load_save_folder
#         self.normalization_method = normalization_method
#         self.cutGraph = cutGraphForPatchOutside
#         if self.folder:
#             os.makedirs(self.folder, exist_ok=True)
#
#     def getStruct(self, sampling_res, num_hops, patch_res=None, patch_id=None):
#
#         if (num_hops is None) or (num_hops <= 0):
#             num_hops = 1
#
#         if self.folder:
#             filename = "sdpa_{}_{}_{}_{}_{}_{}".format(self.weight_type, self.normalization_method, self.use_geodesic,
#                                                        self.use_4connectivity, sampling_res, num_hops)
#             if patch_res:
#                 filename = filename + "_{}_{}_{}".format(patch_res, patch_id, self.cutGraph)
#             filename += ".pth"
#             file_address = os.path.join(self.folder, filename)
#             if os.path.isfile(file_address):
#                 # print("Loading file {}".format(file_address))
#                 data_dict = torch.load(file_address)
#                 index = data_dict.get("index", None)
#                 weight = data_dict.get("weight", None)
#                 valid_neighbors = data_dict.get("mask_valid", None)
#                 if patch_res is None:
#                     return index, weight, valid_neighbors
#                 nodes = data_dict.get("nodes", None)
#                 mapping = data_dict.get("mapping", None)
#                 return index, weight, valid_neighbors, nodes, mapping
#
#         # the major part of the useful code
#         if patch_res is None:
#             nside = hp.order2nside(sampling_res)  # == 2 ** sampling_resolution
#             nPix = hp.nside2npix(nside)
#             pixel_id = np.arange(0, nPix, dtype=int)
#
#             index, weight, valid_neighbors = hp_utils.k_hop_healpix_weightmatrix(resolution=sampling_res,
#                                                                                  weight_type=self.weight_type,
#                                                                                  use_geodesic=self.use_geodesic,
#                                                                                  use_4=self.use_4connectivity,
#                                                                                  nodes_id=pixel_id,
#                                                                                  dtype=np.float32,
#                                                                                  nest=self.isNest,
#                                                                                  num_hops=num_hops,
#                                                                                  )
#
#             index, weight = self.__normalize(index, weight, valid_neighbors, self.normalization_method)
#
#             if self.folder:
#                 print("Saving file {}".format(file_address))
#                 torch.save({"index": index, "weight": weight, "mask_valid": valid_neighbors}, file_address)
#
#             return index, weight, valid_neighbors
#
#         # for Patch based, we temporary deactivate normalization for the whole images_equirectangular because we want to have the normalization per patch
#         tmp_norm = self.normalization_method
#         self.normalization_method = "non"
#         index, weight, valid_neighbors = self.getStruct(sampling_res=sampling_res, num_hops=num_hops)
#         self.normalization_method = tmp_norm  # return back to the original normalization
#
#         n_patches, nPix_per_patch = self.getPatchesInfo(sampling_res, patch_res)
#         assert patch_id >= 0 and patch_id < n_patches, "patch_id={} is not in valid range [0, {})".format(patch_id,
#                                                                                                           n_patches)
#
#         # https://github.com/rusty1s/pytorch_geometric/issues/1205
#         # https://github.com/rusty1s/pytorch_geometric/issues/973
#         interested_nodes = torch.arange(nPix_per_patch * patch_id, nPix_per_patch * (patch_id + 1), dtype=torch.long)
#
#         if self.cutGraph:
#             index = index.narrow(dim=0, start=nPix_per_patch * patch_id, length=nPix_per_patch).detach().clone()
#             weight = weight.narrow(dim=0, start=nPix_per_patch * patch_id, length=nPix_per_patch).detach().clone()
#             valid_neighbors = (index >= nPix_per_patch * patch_id) & (
#                     index < nPix_per_patch * (patch_id + 1)).detach().clone()
#             index -= nPix_per_patch * patch_id
#
#             nodes = interested_nodes
#             mapping = None
#         else:
#             tmp_valid = valid_neighbors.narrow(dim=0, start=nPix_per_patch * patch_id,
#                                                length=nPix_per_patch).clone().detach()
#             nodes, inv = index.narrow(dim=0, start=nPix_per_patch * patch_id, length=nPix_per_patch)[tmp_valid].unique(
#                 return_inverse=True)
#             mapping = (nodes.unsqueeze(1) == interested_nodes).nonzero()[:, 0]
#             index = index.index_select(dim=0, index=nodes)
#             weight = weight.index_select(dim=0, index=nodes)
#             valid_neighbors = torch.zeros(len(nodes), valid_neighbors.size(1), dtype=torch.bool)
#             valid_neighbors[mapping, :] = tmp_valid
#             index[valid_neighbors] = inv
#
#         # print("before=", weight[:10, :])
#         # print("valid neighbor before=", valid_neighbors[:10, :])
#         index, weight = self.__normalize(index, weight, valid_neighbors, self.normalization_method)
#         # print("after=", weight[:10, :])
#         # print("valid neighbor after=", valid_neighbors[:10, :])
#         # index[~valid_neighbors] = 0
#         # weight[~valid_neighbors] = 0
#
#         if self.folder:
#             print("Saving file {}".format(file_address))
#             torch.save({"index": index,
#                         "weight": weight,
#                         "mask_valid": valid_neighbors,
#                         "nodes": nodes,
#                         "mapping": mapping},
#                        file_address)
#         return index, weight, valid_neighbors, nodes, mapping
#
#     def __normalize(self, index, weight, valid_neighbors, normalization_method):
#         assert normalization_method in ['non', 'sym', "sym8", 'sym_neighbors',
#                                         'global_directional_avg'], 'normalization_method not defined'
#
#         if not isinstance(index, torch.Tensor):
#             index = torch.from_numpy(index)
#         if not isinstance(weight, torch.Tensor):
#             weight = torch.from_numpy(weight)
#         if not isinstance(valid_neighbors, torch.Tensor):
#             valid_neighbors = torch.from_numpy(valid_neighbors)
#
#         index[~valid_neighbors] = 0
#         weight[~valid_neighbors] = 0
#
#         if normalization_method == "non":
#             return index, weight
#
#         if normalization_method == "sym":
#             weight.div_(weight.sum(dim=1, keepdim=True))
#         elif normalization_method == "sym8":
#             weight.div_(weight.sum(dim=1, keepdim=True))
#             weight *= 8
#         elif normalization_method == "sym_neighbors":
#             n_neighbors = valid_neighbors.sum(dim=1, keepdim=True)
#             weight.div_(weight.sum(dim=1, keepdim=True))
#             weight.mul_(n_neighbors)
#         elif normalization_method == "global_directional_avg":
#             for col in range(weight.shape[1]):
#                 weight_col = weight[:, col]
#                 weight_col.div_(weight_col.sum())
#                 if self.weight_type == "distance":
#                     weight_col = 2. - weight_col
#                     raise NotImplementedError("Not sure about it")
#
#         return index, weight
#
#     def getPatchesInfo(self, sampling_res, patch_res):
#         assert patch_res <= sampling_res, "patch_res can not be greater than sampling_res"
#         nside = hp.order2nside(sampling_res)  # == 2 ** sampling_resolution
#
#         if patch_res is None or patch_res < 0:  # Negative value means that the whole sphere is desired
#             return 1, hp.nside2npix(nside)
#
#         patch_width = hp.order2nside(patch_res)
#         nPix_per_patch = patch_width * patch_width
#         nside_patch = nside // patch_width
#         n_patches = hp.nside2npix(nside_patch)
#         return n_patches, nPix_per_patch
#
#     def getLayerStructUpsampling(self, scaling_factor_upsampling, hop_upsampling, resolution, patch_resolution=None,
#                                  patch_id=None, inputHopFromDownsampling=None):
#         # print("starting unsampling graph construction", flush=True)
#         assert len(scaling_factor_upsampling) == len(
#             hop_upsampling), "list size for scaling factor and hop numbers must be equal"
#         nconv_layers = len(scaling_factor_upsampling)
#         list_sampling_res_conv, list_patch_res_conv = [[None] * nconv_layers for i in range(2)]
#         list_sampling_res_conv[0] = resolution
#         list_patch_res_conv[0] = patch_resolution
#
#         patching = False
#         if all(v is not None for v in [patch_resolution, patch_id]) and (patch_resolution > 0):
#             patching = True
#
#         for l in range(1, nconv_layers):
#             list_sampling_res_conv[l] = hp_utils.healpix_getResolutionUpsampled(list_sampling_res_conv[l - 1],
#                                                                                 scaling_factor_upsampling[l - 1])
#             if patching:
#                 list_patch_res_conv[l] = hp_utils.healpix_getResolutionUpsampled(list_patch_res_conv[l - 1],
#                                                                                  scaling_factor_upsampling[l - 1])
#
#         highest_sampling_res = hp_utils.healpix_getResolutionUpsampled(list_sampling_res_conv[-1],
#                                                                        scaling_factor_upsampling[-1])
#         if patching:
#             highest_patch_res = hp_utils.healpix_getResolutionUpsampled(list_patch_res_conv[-1],
#                                                                         scaling_factor_upsampling[-1])
#
#         list_index, list_weight, list_mapping_upsampling = [[None] * nconv_layers for i in range(3)]
#
#         K = hop_upsampling.copy()
#         if inputHopFromDownsampling is not None:
#             K[0] += inputHopFromDownsampling
#
#         if not patching:
#             l_first = next(
#                 (i for i in reversed(range(nconv_layers)) if list_sampling_res_conv[-1] != list_sampling_res_conv[i]),
#                 -1) + 1
#             aggregated_K = np.sum(K[
#                                   l_first:])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
#             index, weight, _ = self.getStruct(sampling_res=list_sampling_res_conv[-1], num_hops=aggregated_K)
#             list_index[l_first], list_weight[l_first] = index, weight
#             for l in reversed(range(nconv_layers - 1)):
#                 if list_sampling_res_conv[l] != list_sampling_res_conv[l + 1]:
#                     l_first = next(
#                         (i for i in reversed(range(l + 1)) if list_sampling_res_conv[l] != list_sampling_res_conv[i]),
#                         -1) + 1
#                     aggregated_K = np.sum(K[
#                                           l_first:l + 1])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
#                     index, weight, _ = self.getStruct(sampling_res=list_sampling_res_conv[l], num_hops=aggregated_K)
#
#                     list_index[l_first], list_weight[l_first] = index, weight
#
#             return {"list_sampling_res": list_sampling_res_conv, "list_index": list_index, "list_weight": list_weight,
#                     "output_sampling_res": highest_sampling_res}
#
#         if self.cutGraph:  # cutting the graph in the patch part. This means that border nodes lose their connectivity with outside of the patch
#             l_first = next(
#                 (i for i in reversed(range(nconv_layers)) if list_sampling_res_conv[-1] != list_sampling_res_conv[i]),
#                 -1) + 1
#             aggregated_K = np.sum(K[
#                                   l_first:])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
#             index, weight, _, _, _ = self.getStruct(sampling_res=list_sampling_res_conv[-1], num_hops=aggregated_K,
#                                                     patch_res=list_patch_res_conv[-1], patch_id=patch_id)
#             list_index[l_first], list_weight[l_first] = index, weight
#             for l in reversed(range(nconv_layers - 1)):
#                 if list_sampling_res_conv[l] != list_sampling_res_conv[l + 1]:
#                     l_first = next(
#                         (i for i in reversed(range(l + 1)) if list_sampling_res_conv[l] != list_sampling_res_conv[i]),
#                         -1) + 1
#                     aggregated_K = np.sum(K[
#                                           l_first:l + 1])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
#                     index, weight, _, _, _ = self.getStruct(sampling_res=list_sampling_res_conv[l],
#                                                             num_hops=aggregated_K, patch_res=list_patch_res_conv[l],
#                                                             patch_id=patch_id)
#                     list_index[l_first], list_weight[l_first] = index, weight
#
#             return {"list_sampling_res": list_sampling_res_conv, "list_patch_res": list_patch_res_conv,
#                     "list_index": list_index, "list_weight": list_weight,
#                     "output_sampling_res": highest_sampling_res, "output_patch_res": highest_patch_res}
#
#         # TODO: This part has not been checked for bugs
#         l_first = next(
#             (i for i in reversed(range(nconv_layers)) if list_sampling_res_conv[-1] != list_sampling_res_conv[i]),
#             -1) + 1
#         aggregated_K = np.sum(
#             K[l_first:])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
#         index, weight, _, nodes, mapping = self.getStruct(sampling_res=list_sampling_res_conv[-1],
#                                                           num_hops=aggregated_K, patch_res=list_patch_res_conv[-1],
#                                                           patch_id=patch_id)
#
#         if highest_sampling_res != list_sampling_res_conv[-1]:
#             n_bitshit = 2 * (highest_sampling_res - list_sampling_res_conv[-1])
#             n_children = 1 << n_bitshit
#             mapping = mapping << n_bitshit
#             mapping = mapping.unsqueeze(1).repeat(1, n_children) + torch.arange(n_children)
#             mapping = mapping.flatten()
#         list_mapping_upsampling[-1] = mapping
#         list_index[l_first], list_weight[l_first] = index, weight
#
#         for l in reversed(range(nconv_layers - 1)):
#             if list_sampling_res_conv[l] != list_sampling_res_conv[l + 1]:
#                 n_bitshit = 2 * (list_sampling_res_conv[l + 1] - list_sampling_res_conv[l])
#                 parent_nodes = nodes >> n_bitshit
#                 parent_nodes = parent_nodes.unique()
#
#                 l_first = next(
#                     (i for i in reversed(range(l + 1)) if list_sampling_res_conv[l] != list_sampling_res_conv[i]),
#                     -1) + 1
#                 aggregated_K = np.sum(K[
#                                       l_first:l + 1])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
#                 index, weight, valid_neighbors = self.getStruct(sampling_res=list_sampling_res_conv[l],
#                                                                 num_hops=aggregated_K)
#
#                 index = index.index_select(0, parent_nodes)
#                 weight = weight.index_select(0, parent_nodes)
#                 valid_neighbors = valid_neighbors.index_select(0, parent_nodes)
#
#                 parent_nodes, inv = index[valid_neighbors].unique(return_inverse=True)
#                 index[valid_neighbors] = inv
#
#                 index[~valid_neighbors] = 0
#                 weight[~valid_neighbors] = 0
#
#                 n_children = 1 << n_bitshit
#                 generated_children_nodes_next_layer = parent_nodes << n_bitshit
#                 generated_children_nodes_next_layer = generated_children_nodes_next_layer.unsqueeze(1).repeat(1,
#                                                                                                               n_children) + torch.arange(
#                     n_children)
#                 generated_children_nodes_next_layer = generated_children_nodes_next_layer.flatten()
#                 mapping = (nodes.unsqueeze(1) == generated_children_nodes_next_layer).nonzero()[:, 1]
#
#                 nodes = parent_nodes
#
#                 list_mapping_upsampling[l] = mapping
#                 list_index[l_first], list_weight[l_first] = index, weight
#
#         # print("ending unsampling graph construction", flush=True)
#         return {"list_sampling_res": list_sampling_res_conv, "list_patch_res": list_patch_res_conv,
#                 "list_index": list_index, "list_weight": list_weight,
#                 "list_mapping": list_mapping_upsampling,
#                 "input_nodes": nodes,
#                 "output_sampling_res": highest_sampling_res, "output_patch_res": highest_patch_res}
#
#     def getLayerStructs(self, scaling_factor_downsampling, hop_downsampling, scaling_factor_upsampling, hop_upsampling,
#                         upsampled_resolution, patch_upsampled_resolution=None, patch_id=None):
#
#         assert len(scaling_factor_downsampling) == len(
#             hop_downsampling), "number of layers between scale factor and hops must be equal"
#         nlayers_downsampling = len(scaling_factor_downsampling)
#
#         assert len(scaling_factor_upsampling) == len(
#             hop_upsampling), "number of layers between scale factor and hops must be equal"
#
#         patching = False
#         if all(v is not None for v in [patch_upsampled_resolution, patch_id]) and (patch_upsampled_resolution > 0):
#             patching = True
#
#         list_downsampling_res_conv, list_downsampling_patch_res_conv = [[None] * nlayers_downsampling for i in range(2)]
#         list_downsampling_res_conv[0] = upsampled_resolution
#         list_downsampling_patch_res_conv[0] = patch_upsampled_resolution
#
#         for l in range(1, nlayers_downsampling):
#             list_downsampling_res_conv[l] = hp_utils.healpix_getResolutionDownsampled(list_downsampling_res_conv[l - 1],
#                                                                                       scaling_factor_downsampling[
#                                                                                           l - 1])
#             if patching:
#                 list_downsampling_patch_res_conv[l] = hp_utils.healpix_getResolutionDownsampled(
#                     list_downsampling_patch_res_conv[l - 1], scaling_factor_downsampling[l - 1])
#
#         lowest_sampling_res = hp_utils.healpix_getResolutionDownsampled(list_downsampling_res_conv[-1],
#                                                                         scaling_factor_downsampling[-1])
#         if patching:
#             lowest_patch_res = hp_utils.healpix_getResolutionDownsampled(list_downsampling_patch_res_conv[-1],
#                                                                          scaling_factor_downsampling[-1])
#
#         list_index_downsampling, list_weight_downsampling, list_mapping_downsampling = [[None] * nlayers_downsampling
#                                                                                         for i in range(3)]
#
#         lowest_res_aggregated_hop = 0
#         if list_downsampling_res_conv[-1] == lowest_sampling_res:
#             l_first = next((i for i in reversed(range(nlayers_downsampling)) if
#                             list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
#             lowest_res_aggregated_hop = np.sum(hop_downsampling[l_first:])
#
#         if not patching:
#             dict_graphs = dict()
#             dict_graphs["upsampling"] = self.getLayerStructUpsampling(scaling_factor_upsampling, hop_upsampling,
#                                                                       lowest_sampling_res,
#                                                                       inputHopFromDownsampling=lowest_res_aggregated_hop)
#             l_first = next((i for i in reversed(range(nlayers_downsampling)) if
#                             list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
#             if list_downsampling_res_conv[-1] == lowest_sampling_res:
#                 index = dict_graphs["upsampling"]["list_index"][0]
#                 weight = dict_graphs["upsampling"]["list_weight"][0]
#             else:
#                 aggregated_K = np.sum(hop_downsampling[
#                                       l_first:])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
#                 index, weight, _ = self.getStruct(sampling_res=list_downsampling_res_conv[-1], num_hops=aggregated_K)
#
#             list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight
#             for l in reversed(range(nlayers_downsampling - 1)):
#                 if list_downsampling_res_conv[l] != list_downsampling_res_conv[l + 1]:
#                     l_first = next((i for i in reversed(range(l + 1)) if
#                                     list_downsampling_res_conv[l] != list_downsampling_res_conv[i]), -1) + 1
#                     aggregated_K = np.sum(hop_downsampling[
#                                           l_first:l + 1])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
#                     index, weight, _ = self.getStruct(sampling_res=list_downsampling_res_conv[l], num_hops=aggregated_K)
#                     list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight
#
#             dict_graphs["downsampling"] = {"list_sampling_res": list_downsampling_res_conv,
#                                            "list_index": list_index_downsampling,
#                                            "list_weight": list_weight_downsampling}
#             return dict_graphs
#
#         if self.cutGraph:  # cutting the graph in the patch part. This means that border nodes lose their connectivity with outside of the patch
#             dict_graphs = dict()
#             dict_graphs["upsampling"] = self.getLayerStructUpsampling(scaling_factor_upsampling, hop_upsampling,
#                                                                       lowest_sampling_res,
#                                                                       patch_resolution=lowest_patch_res,
#                                                                       patch_id=patch_id,
#                                                                       inputHopFromDownsampling=lowest_res_aggregated_hop)
#             l_first = next((i for i in reversed(range(nlayers_downsampling)) if
#                             list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
#             if list_downsampling_res_conv[-1] == lowest_sampling_res:
#                 index = dict_graphs["upsampling"]["list_index"][0]
#                 weight = dict_graphs["upsampling"]["list_weight"][0]
#             else:
#                 aggregated_K = np.sum(hop_downsampling[
#                                       l_first:])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
#                 index, weight, _, _, _ = self.getStruct(sampling_res=list_downsampling_res_conv[-1],
#                                                         num_hops=aggregated_K,
#                                                         patch_res=list_downsampling_patch_res_conv[-1],
#                                                         patch_id=patch_id)
#
#             list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight
#             for l in reversed(range(nlayers_downsampling - 1)):
#                 if list_downsampling_res_conv[l] != list_downsampling_res_conv[l + 1]:
#                     l_first = next((i for i in reversed(range(l + 1)) if
#                                     list_downsampling_res_conv[l] != list_downsampling_res_conv[i]), -1) + 1
#                     aggregated_K = np.sum(hop_downsampling[
#                                           l_first:l + 1])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
#                     index, weight, _, _, _ = self.getStruct(sampling_res=list_downsampling_res_conv[l],
#                                                             num_hops=aggregated_K,
#                                                             patch_res=list_downsampling_patch_res_conv[l],
#                                                             patch_id=patch_id)
#                     list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight
#
#             _, nPixPerPatch = self.getPatchesInfo(upsampled_resolution, patch_upsampled_resolution)
#             range_downsampling_input_to_patch = (int(patch_id * nPixPerPatch), int((patch_id + 1) * nPixPerPatch))
#
#             dict_graphs["downsampling"] = {"list_sampling_res": list_downsampling_res_conv,
#                                            "list_patch_res": list_downsampling_patch_res_conv,
#                                            "list_index": list_index_downsampling,
#                                            "list_weight": list_weight_downsampling,
#                                            "range_downsampling_input_to_patch": range_downsampling_input_to_patch}
#             return dict_graphs
#
#         # TODO: This part has not been checked for bugs
#         dict_graphs = dict()
#         dict_graphs["upsampling"] = self.getLayerGraphUpsampling(scaling_factor_upsampling, hop_upsampling,
#                                                                  lowest_sampling_res, patch_resolution=lowest_patch_res,
#                                                                  patch_id=patch_id,
#                                                                  inputHopFromDownsampling=lowest_res_aggregated_hop)
#
#         # print("starting downsampling graph construction", flush=True)
#
#         nodes = dict_graphs["upsampling"]["input_nodes"]
#         index = dict_graphs["upsampling"]["list_index"][0]
#         weight = dict_graphs["upsampling"]["list_weight"][0]
#
#         _, nPixPerPatch = self.getPatchesInfo(lowest_sampling_res, lowest_patch_res)
#         ind_start = (
#                 nodes == patch_id * nPixPerPatch).nonzero().item()  # to find index of the node==patch_id*nPixPerPatch
#         # Maybe later I can remove the next assert check.
#         assert torch.all(torch.eq(nodes.narrow(dim=0, start=ind_start, length=nPixPerPatch),
#                                   torch.arange(patch_id * nPixPerPatch, (patch_id + 1) * nPixPerPatch,
#                                                dtype=nodes.dtype))), "patch nodes from upsampling must already contains last resolution patch nodes in a sorted order"
#         range_downsampling_output_to_patch = (ind_start, ind_start + nPixPerPatch)
#
#         if list_downsampling_res_conv[
#             -1] == lowest_sampling_res:  # This means that last conv layer of downsampling has same size of first conv layer of upsampling
#             l_first = next((i for i in reversed(range(nlayers_downsampling)) if
#                             list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
#             list_mapping_downsampling[
#                 -1] = None  # This means that we are in the middle of layer so no mapping is needed
#             list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight
#         else:
#             n_bitshit = 2 * (list_downsampling_res_conv[-1] - lowest_sampling_res)
#             n_children = 1 << n_bitshit
#             interested_nodes = nodes << n_bitshit
#             interested_nodes = interested_nodes.unsqueeze(1).repeat(1, n_children) + torch.arange(n_children)
#             interested_nodes = interested_nodes.flatten()
#
#             l_first = next((i for i in reversed(range(nlayers_downsampling)) if
#                             list_downsampling_res_conv[-1] != list_downsampling_res_conv[i]), -1) + 1
#             aggregated_K = np.sum(hop_downsampling[
#                                   l_first:])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
#             index, weight, valid_neighbors = self.getStruct(sampling_res=list_downsampling_res_conv[-1],
#                                                             num_hops=aggregated_K)
#
#             index = index.index_select(0, interested_nodes)
#             weight = weight.index_select(0, interested_nodes)
#             valid_neighbors = valid_neighbors.index_select(0, interested_nodes)
#
#             nodes, inv = index[valid_neighbors].unique(return_inverse=True)
#             index[valid_neighbors] = inv
#             mapping = (nodes.unsqueeze(1) == interested_nodes).nonzero()[:, 0]
#
#             index[~valid_neighbors] = 0
#             weight[~valid_neighbors] = 0
#
#             interested_nodes = nodes
#             list_mapping_downsampling[-1] = mapping
#             list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight
#
#         for l in reversed(range(nlayers_downsampling - 1)):
#             if list_downsampling_res_conv[l] != list_downsampling_res_conv[l + 1]:
#                 n_bitshit = 2 * (list_downsampling_res_conv[l] - list_downsampling_res_conv[l + 1])
#                 n_children = 1 << n_bitshit
#                 nodes = nodes << n_bitshit
#                 interested_nodes = interested_nodes.unsqueeze(1).repeat(1, n_children) + torch.arange(n_children)
#                 interested_nodes = interested_nodes.flatten()
#
#                 l_first = next((i for i in reversed(range(l + 1)) if
#                                 list_downsampling_res_conv[l] != list_downsampling_res_conv[i]), -1) + 1
#                 aggregated_K = np.sum(hop_downsampling[
#                                       l_first:l + 1])  # casacde of conv layers at the same resolution has an effective hop equal to sum of each hop
#                 index, weight, valid_neighbors = self.getGraph(sampling_res=list_downsampling_res_conv[l],
#                                                                num_hops=aggregated_K)
#
#                 index = index.index_select(0, interested_nodes)
#                 weight = weight.index_select(0, interested_nodes)
#                 valid_neighbors = valid_neighbors.index_select(0, interested_nodes)
#
#                 nodes, inv = index[valid_neighbors].unique(return_inverse=True)
#                 index[valid_neighbors] = inv
#                 mapping = (nodes.unsqueeze(1) == interested_nodes).nonzero()[:, 0]
#
#                 index[~valid_neighbors] = 0
#                 weight[~valid_neighbors] = 0
#
#                 interested_nodes = nodes
#
#                 list_mapping_downsampling[l] = mapping
#                 list_index_downsampling[l_first], list_weight_downsampling[l_first] = index, weight
#
#         _, nPixPerPatch = self.getPatchesInfo(upsampled_resolution, patch_upsampled_resolution)
#         ind_start = (
#                 nodes == patch_id * nPixPerPatch).nonzero().item()  # to find index of the node==patch_id*nPixPerPatch
#         # Maybe later I can remove the next assert check.
#         assert torch.all(torch.eq(nodes.narrow(dim=0, start=ind_start, length=nPixPerPatch),
#                                   torch.arange(patch_id * nPixPerPatch, (patch_id + 1) * nPixPerPatch,
#                                                dtype=nodes.dtype))), "patch nodes from upsampling must already contains last resolution patch nodes in a sorted order"
#         range_downsampling_input_to_patch = (ind_start, ind_start + nPixPerPatch)
#
#         # print("ending downsampling graph construction", flush=True)
#         dict_graphs["downsampling"] = {"list_sampling_res": list_downsampling_res_conv,
#                                        "list_patch_res": list_downsampling_patch_res_conv,
#                                        "list_index": list_index_downsampling, "list_weight": list_weight_downsampling,
#                                        "input_nodes": nodes, "list_mapping": list_mapping_downsampling,
#                                        "range_downsampling_output_to_patch": range_downsampling_output_to_patch,
#                                        "range_downsampling_input_to_patch": range_downsampling_input_to_patch}
#
#         return dict_graphs

if __name__ == "__main__":
    dvc = DVC()
    # neighbors_indices, neighbors_weights, valid_neighbors = getHEALPixNeighbours()
    # neighbors_indices = neighbors_indices.to('cuda:3')
    # neighbors_weights = neighbors_weights.to('cuda:3')
    # dvc.residual_encoder = (
    #     DVCResidualEncoder(
    #         neighbors_indices=neighbors_indices,
    #         neighbors_weights=neighbors_weights,
    #         in_channels=3,
    #         out_channels=128,
    #         filter_channels=128,
    #         # kernel_size = neighborhood_size + 1
    #         kernel_size=9,
    #
    #     )
    # )
    # dvc.residual_decoder = (
    #     DVCResidualDecoder(
    #         neighbors_indices=neighbors_indices,
    #         neighbors_weights=neighbors_weights,
    #         in_channels=128,
    #         filter_channels=128,
    #         out_channels=3,
    #         # kernel_size=5,
    #         kernel_size=9,
    #     )
    # )
    # image1 = torch.rand((4, 3, 64, 128))
    # image2 = torch.rand((4, 3, 64, 128))
    #
    # dvc.forward(image1, image2)
