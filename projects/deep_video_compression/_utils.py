# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Callable, NamedTuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from torch import Tensor

import neuralcompression.functional as ncF
from neuralcompression.functional import optical_flow_to_color


class LossFunctions(NamedTuple):
    distortion_fn: Callable[[Tensor, Tensor], Tensor]
    entropy_fn: Optional[Callable[[Tensor, int], Tensor]]


class LossValues(NamedTuple):
    distortion_loss: Tensor
    flow_entropy_loss: Optional[Tensor]
    resid_entropy_loss: Optional[Tensor]


class OutputTensors(NamedTuple):
    flow: Optional[Tensor]
    image1: Optional[Tensor]
    image2: Optional[Tensor]
    image2_est: Optional[Tensor]


class DvcStage(nn.Module):
    def collect_parameters(self):
        model_parameters = {
            n: p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        }
        quantile_parameters = {
            n: p
            for n, p in self.named_parameters()
            if n.endswith(".quantiles") and p.requires_grad
        }

        return model_parameters, quantile_parameters

    def forward(self, image1, image2):
        raise NotImplementedError

    def compute_batch_loss(self, image1, image2):
        raise NotImplementedError

    def quantile_loss(self):
        return torch.tensor(0.0)

    def update(self, force=False):
        pass

    def recompose_model(self, model):
        raise NotImplementedError


class DvcStage1(DvcStage):
    def __init__(self, model, num_pframes, loss_functions):
        super().__init__()
        self.model = model.motion_estimator
        self.num_pframes = num_pframes
        self.loss_functions = loss_functions

    def collect_parameters(self):
        model_parameters = {
            n: p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        }
        return model_parameters, None

    def forward(self, image1, image2):
        return self.model(image1, image2)

    def compute_batch_loss(self, image1, image2):
        flow, outputs = self.model.calculate_flow_with_image_pairs(image1, image2)

        frame_losses = []
        for output in outputs:
            frame_losses.append(self.loss_functions.distortion_fn(output[1], output[0]))

        loss = torch.stack(frame_losses).mean()

        image2_est = ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1).detach())

        return (
            LossValues(
                distortion_loss=loss, flow_entropy_loss=None, resid_entropy_loss=None
            ),
            OutputTensors(
                flow=flow.detach(),
                image1=image1.detach(),
                image2=image2.detach(),
                image2_est=image2_est.detach(),
            ),
        )

    def recompose_model(self, model):
        model.motion_estimator = self.model
        return model


class DvcStage2(DvcStage):
    def __init__(
        self,
        model,
        num_pframes,
        loss_functions: LossFunctions,
    ):
        super().__init__()
        self.motion_estimator = model.motion_estimator
        self.motion_encoder = model.motion_encoder
        self.motion_entropy_bottleneck = model.motion_entropy_bottleneck
        self.motion_decoder = model.motion_decoder
        self.num_pframes = num_pframes
        if loss_functions.entropy_fn is None:
            raise ValueError("Must specify entropy_fn for compression stage.")
        self.loss_functions = loss_functions


    def collect_parameters(self):
        model_parameters = {
            n: p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        }



        return model_parameters, None

    def forward(self, image1, image2):
        flow = self.motion_estimator(image1, image2)
        ori_flow = flow
        latent, sizes = self.motion_encoder(flow)
        latent, probabilities = self.motion_entropy_bottleneck(latent)
        flow = self.motion_decoder(latent, sizes)
        image2_est = ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1))

        # plot flow
        # Compute the figure size
        # fig_height_per_subplot = 256 / 80  # assuming 80 dpi, you can adjust this
        # fig_width_per_subplot = 448 / 80  # assuming 80 dpi, you can adjust this
        # fig_width = 2 * fig_width_per_subplot
        # fig_height = fig_height_per_subplot

        # Create a new figure
        # fig, axes = plt.subplots(2, 2)
        #
        #
        # # third_channel = 0 * np.ones((256, 448))
        # ori_flow_plot = torch.clip(optical_flow_to_color(ori_flow)[0].permute(1, 2, 0), min=0, max=1.0).detach().numpy()
        # # ori_flow_plot = np.concatenate((ori_flow_plot, third_channel[:, :, np.newaxis]), axis=2)
        #
        # flow_plot = torch.clip(optical_flow_to_color(flow)[0].permute(1, 2, 0), min=0, max=1.0).detach().numpy()
        # # flow_plot = np.concatenate((flow_plot, third_channel[:, :, np.newaxis]), axis=2)
        #
        # axes[0, 0].imshow(image2.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
        # axes[0, 0].text(0.5, -0.2, 'Reconstructed Frame', ha='center', va='center', transform=axes[0, 0].transAxes)
        # axes[0, 0].axis('off')
        # axes[0, 1].imshow(image2_est.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
        # axes[0, 1].text(0.5, -0.2, 'Current Frame', ha='center', va='center', transform=axes[0, 1].transAxes)
        # axes[0, 1].axis('off')
        #
        # axes[1, 0].imshow(ori_flow_plot)
        # axes[1, 0].text(0.5, -0.2, 'Optical Flow', ha='center', va='center', transform=axes[1, 0].transAxes)
        # axes[1, 0].axis('off')
        #
        # axes[1, 1].imshow(flow_plot)
        # axes[1, 1].text(0.5, -0.2, 'Reconstructed Optical Flow', ha='center', va='center', transform=axes[1, 1].transAxes)
        # axes[1, 1].axis('off')
        #
        # prefix = 'performance'
        # path = prefix + '/compare-flow' + '.png'
        # if not os.path.exists(prefix):
        #     os.makedirs(prefix)
        # plt.savefig(path)
        # plt.clf()
        #
        # image1_save = (image1.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy() * 255).astype(np.uint8)
        # cv2.imwrite("performance/reconstructed-frame.png", cv2.cvtColor(image1_save, cv2.COLOR_RGB2BGR))
        # image2_save = (image2.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy() * 255).astype(np.uint8)
        # cv2.imwrite("performance/current-frame.png", cv2.cvtColor(image2_save, cv2.COLOR_RGB2BGR))
        # ori_flow_save = (ori_flow_plot * 255).astype(np.uint8)
        # cv2.imwrite("performance/optical-flow.png", cv2.cvtColor(ori_flow_save, cv2.COLOR_RGB2BGR))
        # flow_save = (flow_plot * 255).astype(np.uint8)
        # cv2.imwrite("performance/reconstructed-optical-flow.png", cv2.cvtColor(flow_save, cv2.COLOR_RGB2BGR))
        #
        # # Calculate Magnitude
        # magnitude = np.sqrt(np.sum(ori_flow_save ** 2, axis=2)).flatten()
        #
        # # Calculate Histogram
        # hist, bins = np.histogram(magnitude, bins=256)
        #
        # # Normalize to get probability distribution
        # hist = hist / hist.sum()
        #
        # # Plot
        # plt.figure()
        # plt.bar(bins[:-1], hist, width=np.diff(bins),  color='dodgerblue', alpha=0.7)
        # plt.grid(True, linestyle='--', alpha=0.7)
        # plt.xlabel("Pixel Magnitude", fontsize=14)
        # plt.ylabel("Probability", fontsize=14)
        # step = 0.002
        # y_ticks = np.arange(0, 0.02 + step, step)
        # plt.ylim(0, 0.02)
        # plt.yticks(y_ticks)
        # # Set the yticks to display 2 decimal places
        # y_formatter = FormatStrFormatter('%.3f')
        # plt.gca().yaxis.set_major_formatter(y_formatter)
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        # # Save the plot to a file
        # plt.savefig("optical-flow-distribution.png")
        #
        # # Calculate Magnitude
        # magnitude = np.sqrt(np.sum(flow_save ** 2, axis=2)).flatten()
        #
        # # Calculate Histogram
        # hist, bins = np.histogram(magnitude, bins=256)
        #
        # # Normalize to get probability distribution
        # hist = hist / hist.sum()
        #
        # # Plot
        # plt.figure()
        # plt.bar(bins[:-1], hist, width=np.diff(bins),  color='dodgerblue', alpha=0.7)
        # plt.grid(True, linestyle='--', alpha=0.7)
        # plt.xlabel("Pixel Magnitude", fontsize=14)
        # plt.ylabel("Probability", fontsize=14)
        # step = 0.002
        # y_ticks = np.arange(0, 0.02 + step, step)
        # plt.ylim(0, 0.02)
        # plt.yticks(y_ticks)
        # # Set the yticks to display 2 decimal places
        # y_formatter = FormatStrFormatter('%.3f')
        # plt.gca().yaxis.set_major_formatter(y_formatter)
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        #
        # # Save the plot to a file
        # plt.savefig("reconstructed-flow-distribution.png")

        return flow, image2_est, probabilities

    def compute_batch_loss(self, image1, image2):
        assert self.loss_functions.entropy_fn is not None
        assert image1.ndim == image2.ndim == 4

        flow, image2_est, probabilities = self.forward(image1, image2)

        # compute distortion loss
        distortion_loss = self.loss_functions.distortion_fn(image2, image2_est)

        # compute compression loss, average over num pixels
        num_pixels = image1.shape[0] * image1.shape[-2] * image1.shape[-1]
        entropy_loss = self.loss_functions.entropy_fn(probabilities, num_pixels)

        return (
            LossValues(
                distortion_loss=distortion_loss,
                flow_entropy_loss=entropy_loss,
                resid_entropy_loss=None,
            ),
            OutputTensors(
                flow=flow.detach(),
                image1=image1.detach(),
                image2=image2.detach(),
                image2_est=image2_est.detach(),
            ),
        )

    def quantile_loss(self):
        return self.motion_entropy_bottleneck.loss()

    def update(self, force=False):
        return self.motion_entropy_bottleneck.update(force=force)

    def recompose_model(self, model):
        model.motion_estimator = self.motion_estimator
        model.motion_encoder = self.motion_encoder
        model.motion_entropy_bottleneck = self.motion_entropy_bottleneck
        model.motion_decoder = self.motion_decoder
        return model


class DvcStage3(DvcStage):
    def __init__(
        self,
        model,
        num_pframes,
        loss_functions: LossFunctions,
    ):
        super().__init__()
        self.motion_estimator = model.motion_estimator
        self.motion_encoder = model.motion_encoder
        self.motion_entropy_bottleneck = model.motion_entropy_bottleneck
        self.motion_decoder = model.motion_decoder
        self.motion_compensation = model.motion_compensation
        self.num_pframes = num_pframes
        if loss_functions.entropy_fn is None:
            raise ValueError("Must specify entropy_fn for compression stage.")
        self.loss_functions = loss_functions


    def collect_parameters(self):
        model_parameters = {
            n: p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        }
        return model_parameters, None

    def forward(self, image1, image2):
        flow = self.motion_estimator(image1, image2)
        ori_flow = flow
        latent, sizes = self.motion_encoder(flow)
        latent, probabilities = self.motion_entropy_bottleneck(latent)
        flow = self.motion_decoder(latent, sizes)
        image2_est = ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1))
        warped_image2_est = image2_est

        image2_est = self.motion_compensation(image1, image2_est, flow)


        # plot flow

        # # Create a new figure
        # fig, axes = plt.subplots(2, 2)
        #
        #
        # # third_channel = 0 * np.ones((256, 448))
        # ori_flow_plot = torch.clip(optical_flow_to_color(ori_flow)[0].permute(1, 2, 0), min=0, max=1.0).detach().numpy()
        # # ori_flow_plot = np.concatenate((ori_flow_plot, third_channel[:, :, np.newaxis]), axis=2)
        #
        # flow_plot = torch.clip(optical_flow_to_color(flow)[0].permute(1, 2, 0), min=0, max=1.0).detach().numpy()
        # # flow_plot = np.concatenate((flow_plot, third_channel[:, :, np.newaxis]), axis=2)
        #
        # axes[0, 0].imshow(image2.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
        # axes[0, 0].text(0.5, -0.2, 'Current Frame', ha='center', va='center', transform=axes[0, 0].transAxes)
        # axes[0, 0].axis('off')
        # axes[0, 1].imshow(image2_est.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
        # axes[0, 1].text(0.5, -0.2, 'Compensated Frame', ha='center', va='center', transform=axes[0, 1].transAxes)
        # axes[0, 1].axis('off')
        #
        # axes[1, 0].imshow(ori_flow_plot)
        # axes[1, 0].text(0.5, -0.2, 'Optical Flow', ha='center', va='center', transform=axes[1, 0].transAxes)
        # axes[1, 0].axis('off')
        #
        # axes[1, 1].imshow(flow_plot)
        # axes[1, 1].text(0.5, -0.2, 'Reconstructed Optical Flow', ha='center', va='center', transform=axes[1, 1].transAxes)
        # axes[1, 1].axis('off')
        #
        # prefix = 'performance'
        # path = prefix + '/compare-flow' + '.png'
        # if not os.path.exists(prefix):
        #     os.makedirs(prefix)
        # plt.savefig(path)
        # plt.clf()

        # image1_save = (image1.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy() * 255).astype(np.uint8)
        # cv2.imwrite("performance/previous-frame-new.png", cv2.cvtColor(image1_save, cv2.COLOR_RGB2BGR))
        # image2_save = (image2.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy() * 255).astype(np.uint8)
        # cv2.imwrite("performance/current-frame-new.png", cv2.cvtColor(image2_save, cv2.COLOR_RGB2BGR))
        # ori_flow_save = (ori_flow_plot * 255).astype(np.uint8)
        # cv2.imwrite("performance/optical-flow-new.png", cv2.cvtColor(ori_flow_save, cv2.COLOR_RGB2BGR))
        # flow_save = (flow_plot * 255).astype(np.uint8)
        # cv2.imwrite("performance/reconstructed-optical-flow-new.png", cv2.cvtColor(flow_save, cv2.COLOR_RGB2BGR))
        # #


        return flow, image2_est, probabilities

    def compute_batch_loss(self, image1, image2):
        assert self.loss_functions.entropy_fn is not None
        assert image1.ndim == image2.ndim == 4

        flow, image2_est, probabilities = self.forward(image1, image2)

        # compute distortion loss
        distortion_loss = self.loss_functions.distortion_fn(image2, image2_est)

        # compute compression loss, average over num pixels
        num_pixels = image1.shape[0] * image1.shape[-2] * image1.shape[-1]
        entropy_loss = self.loss_functions.entropy_fn(probabilities, num_pixels)

        return (
            LossValues(
                distortion_loss=distortion_loss,
                flow_entropy_loss=entropy_loss,
                resid_entropy_loss=None,
            ),
            OutputTensors(
                flow=flow.detach(),
                image1=image1.detach(),
                image2=image2.detach(),
                image2_est=image2_est.detach(),
            ),
        )

    def quantile_loss(self):
        return self.motion_entropy_bottleneck.loss()

    def update(self, force=False):
        return self.motion_entropy_bottleneck.update(force=force)

    def recompose_model(self, model):
        model.motion_estimator = self.motion_estimator
        model.motion_encoder = self.motion_encoder
        model.motion_entropy_bottleneck = self.motion_entropy_bottleneck
        model.motion_decoder = self.motion_decoder
        model.motion_compensation = self.motion_compensation
        return model


class DvcStage4and5(DvcStage):
    def __init__(self, dvc_model, num_pframes, loss_functions: LossFunctions):
        super().__init__()
        self.model = dvc_model
        self.num_pframes = num_pframes
        if loss_functions.entropy_fn is None:
            raise ValueError("Must specify entropy_fn for compression stage.")
        self.loss_functions = loss_functions


    def collect_parameters(self):
        model_parameters = {
            n: p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        }
        return model_parameters, None
    # TODO: modify forward
    def forward(self, image1, image2, dict_index=None, dict_weight=None, sample_res=None, patch_res=None, patch_id=None, nPix_per_patch=None):
        return self.model(image1, image2, dict_index, dict_weight, sample_res, patch_res, patch_id, nPix_per_patch)

    def compute_batch_loss(self, image1, image2, dict_index=None, dict_weight=None, sample_res=None, patch_res=None, patch_id=None, nPix_per_patch=None):
        assert self.loss_functions.entropy_fn is not None
        assert image1.ndim == image2.ndim == 4

        output = self.forward(image1, image2, dict_index, dict_weight, sample_res, patch_res, patch_id, nPix_per_patch)

        # compute distortion loss. TODO: To be discussed
        distortion_loss = self.loss_functions.distortion_fn(image2, output.image2_est)

        # compute flow compression loss, average over num pixels
        num_pixels = image1.shape[0] * image1.shape[-2] * image1.shape[-1]
        flow_entropy_loss = self.loss_functions.entropy_fn(
            output.flow_probabilities, num_pixels
        )

        # compute resid compression loss, average over num pixels
        resid_entropy_loss = self.loss_functions.entropy_fn(
            output.resid_probabilities, num_pixels
            # output.resid_probabilities, num_pixels if nPix_per_patch is None else nPix_per_patch
        )

        return (
            LossValues(
                distortion_loss=distortion_loss,
                flow_entropy_loss=flow_entropy_loss,
                resid_entropy_loss=resid_entropy_loss,
            ),
            OutputTensors(
                flow=output.flow.detach(),
                image1=image1.detach(),
                image2=image2.detach(),
                image2_est=output.image2_est.detach(),
            ),
        )

    def quantile_loss(self):
        return (
            self.model.motion_entropy_bottleneck.loss()
            + self.model.residual_entropy_bottleneck.loss()
        )

    def update(self, force=False) -> bool:
        update1 = self.model.motion_entropy_bottleneck.update(force=force)
        update2 = self.model.residual_entropy_bottleneck.update(force=force)
        return update1 or update2

    def recompose_model(self, model):
        return self.model

