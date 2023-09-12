# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
from typing import NamedTuple, Optional, Tuple
from pytorch_msssim import ms_ssim

import cv2
import healpy as hp
import numpy as np
import torch
import torch.optim as optim
import torchmetrics
import utils.healpix as hp_utils
import time
import matplotlib.pyplot as plt

from _utils import (
    DvcStage,
    DvcStage1,
    DvcStage2,
    DvcStage3,
    DvcStage4and5,
    LossFunctions,
)
from compressai.zoo import models
from pytorch_lightning import LightningModule
from torch import Tensor

import neuralcompression.functional as ncF
from neuralcompression.models import DVC
from projects.deep_video_compression.preprocessing.healpix_sdpa_struct_loader import HealpixSdpaStructLoader

TRAINING_STAGES = {
    "1_motion_estimation": DvcStage1,
    "2_motion_compression": DvcStage2,
    "3_motion_compensation": DvcStage3,
    "4_total_2frame": DvcStage4and5,
    "5_total": DvcStage4and5,
}


class LoggingMetrics(NamedTuple):
    gop_total_loss: Tensor
    gop_distortion_loss: Tensor
    gop_bpp: Tensor
    gop_flow_bpp: Optional[Tensor]
    gop_residual_bpp: Optional[Tensor]


class DvcModule(LightningModule):
    """
    Model and training loop for the DVC model.

    Combines a pre-defined DVC model with its training loop for use with
    PyTorch Lightning.

    Args:
        model: The DVC model to train.
        training_stage: Current stage of training process. One of
            ``("1_motion_estimation", "2_motion_compression",
            "3_motion_compensation", "4_total")``. See DVC paper for details.
        pretrained_model_name: Name of model from CompressAI model zoo to use
            for compressing I-frames.
        pretrained_model_quality_level: Quality level of model from CompressAI.
        num_pframes: Number of P-frames to process for training.
        distortion_type: Type of distortion loss function. Must be from
            ``("MSE")``.
        distortion_lambda: A scaling factor for the distortion term of the
            loss.
        learning_rate: passed to the main network optimizer (i.e. the one that
            adjusts the analysis and synthesis parameters).
        aux_learning_rate: passed to the optimizer that learns the quantiles
            used to build the CDF table for the entropy codder.
        lr_scheduler_params: Used for ``StepLR``, specify ``step_size`` and
            ``gamma``.
    """

    def __init__(
            self,
            model: DVC,
            struct_loader: HealpixSdpaStructLoader,
            training_stage: str,
            pretrained_model_name: str,
            pretrained_model_quality_level: int,
            num_pframes: int = 1,
            distortion_type: str = "MSE",
            distortion_lambda: float = 256.0,
            learning_rate: float = 1e-4,
            aux_learning_rate: float = 1e-3,
            lr_scheduler_params: Optional[Tuple[int, float]] = None,
            grad_clip_value: float = 1.0,
            on_sphere_learning: bool = False,
            patching: bool = True,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.struct_loader = struct_loader
        self.training_stage = training_stage
        self.learning_rate = learning_rate
        self.num_pframes = num_pframes
        self.aux_learning_rate = aux_learning_rate
        self.distortion_lambda = distortion_lambda
        self.lr_scheduler_params = lr_scheduler_params
        self.grad_clip_value = grad_clip_value
        self.iframe_model = models[pretrained_model_name](
            pretrained_model_quality_level, pretrained=True
        )
        self.on_sphere_learning = on_sphere_learning

        # make sure training stage is valid
        if training_stage not in TRAINING_STAGES.keys():
            raise ValueError(f"training stage {training_stage} not recognized.")
        if distortion_type == "MSE":
            distortion_loss = torch.nn.MSELoss()
        else:
            raise ValueError(f"distortion_type {distortion_type} not recognized.")

        # set up loss functions
        def distortion_fn(image1, image2):
            return self.distortion_lambda * distortion_loss(image1, image2)

        def entropy_fn(probabilities, reduction_factor):
            return (
                    ncF.information_content(probabilities, reduction="sum")
                    / reduction_factor
            )

        self.loss_functions = LossFunctions(distortion_fn, entropy_fn)

        # set up model - this includes functions for computing losses and
        # compressing
        self.model: DvcStage = TRAINING_STAGES[training_stage](
            model, num_pframes, self.loss_functions
        )

        # metrics
        self.psnr = torchmetrics.PSNR(data_range=1.0)

        if self.on_sphere_learning:
            sample_res = 7
            patch_res = 4 if patching else None
            self.sample_res = sample_res
            self.patch_res = patch_res
            n_patches, self.nPix_per_patch = struct_loader.getPatchesInfo(sampling_res=sample_res, patch_res=patch_res)
            noPatching = True if n_patches == 1 else False
            self.noPatching = noPatching
            self.healpix_resolution_patch_level = hp.nside2order(hp.npix2nside(n_patches)) if not noPatching else None
            # implement offset
            self.list_res = [sample_res + offset if noPatching else (sample_res + offset, patch_res + offset) for offset
                             in
                             model.residual_network.get_resOffset()]

    def recompose_model(self, model: DVC):
        """Used by trainer to rebuild the model after a training run."""
        return self.model.recompose_model(model)

    def forward(self, images):
        return self.model(images)

    def update(self, force=True):
        return self.model.update(force=force)

    def generate_random_patches(self, patch_level_res, n=1):
        if n == 1:
            n_patches = hp.nside2npix(hp.order2nside(patch_level_res))
            return list(random.sample(range(n_patches), n))

        n_northcap_samples = n_equitorial_samples = n_southcap_samples = n // 3  # We have 3 regions: north cap, equitorial, south cap
        remainder = n % 3

        if remainder:
            n_equitorial_samples += 1
            remainder -= 1

        if remainder:  # if there is still remainder. Now remainder is equal to one
            assert remainder == 1, "the remainder must be equal to one"
            # Randomly sample from north cap or south cap
            val = random.randint(0, 1)
            n_northcap_samples += val
            n_southcap_samples += 1 - val

        list_equitorial = random.sample(
            list(hp_utils.get_regionPixelIds(patch_level_res, "equatorial_region", nest=True)),
            n_equitorial_samples)
        list_north_cap = random.sample(list(hp_utils.get_regionPixelIds(patch_level_res, "north_polar_cap", nest=True)),
                                       n_northcap_samples)
        list_south_cap = random.sample(list(hp_utils.get_regionPixelIds(patch_level_res, "south_polar_cap", nest=True)),
                                       n_southcap_samples)

        list_patch_ids = []
        list_patch_ids.extend(list_equitorial)
        list_patch_ids.extend(list_north_cap)
        list_patch_ids.extend(list_south_cap)
        return list_patch_ids

    def compress_iframe(self, batch):
        """Compress an I-frame using a pretrained model."""
        self.iframe_model = self.iframe_model.eval()
        self.iframe_model.update()
        with torch.no_grad():
            output = self.iframe_model(batch[:, 0])

        batch[:, 0] = output["x_hat"]
        num_pixels = batch.shape[0] * batch.shape[-2] * batch.shape[-1]
        bpp_loss = sum(
            self.loss_functions.entropy_fn(likelihoods, num_pixels)
            for likelihoods in output["likelihoods"].values()
        )

        return batch, bpp_loss

    def compute_loss_and_metrics(self, loss_values, current_metrics):
        # loss function collection
        combined_bpp = 0
        loss = loss_values.distortion_loss
        gop_distortion_loss = (
                current_metrics.gop_distortion_loss + loss_values.distortion_loss.detach()
        )

        if loss_values.flow_entropy_loss is not None:
            loss = loss + loss_values.flow_entropy_loss
            combined_bpp = combined_bpp + loss_values.flow_entropy_loss.detach()
            gop_flow_bpp: Optional[Tensor] = (
                    current_metrics.gop_flow_bpp + loss_values.flow_entropy_loss.detach()
            )
        else:
            gop_flow_bpp = None

        if loss_values.resid_entropy_loss is not None:
            loss = loss + loss_values.resid_entropy_loss
            combined_bpp = combined_bpp + loss_values.resid_entropy_loss.detach()
            gop_residual_bpp: Optional[Tensor] = (
                    current_metrics.gop_residual_bpp
                    + loss_values.resid_entropy_loss.detach()
            )
        else:
            gop_residual_bpp = None

        gop_total_loss = current_metrics.gop_total_loss + loss.detach()
        gop_bpp = current_metrics.gop_bpp + combined_bpp

        return loss, LoggingMetrics(
            gop_total_loss=gop_total_loss,
            gop_distortion_loss=gop_distortion_loss,
            gop_bpp=gop_bpp,
            gop_flow_bpp=gop_flow_bpp,
            gop_residual_bpp=gop_residual_bpp,
        )


    def log_all_metrics(
            self, log_key, reduction, image2_list, image2_est_list, logging_metrics
    ):
        self.log(f"{log_key}gop_loss", logging_metrics.gop_total_loss / reduction)
        self.log(
            f"{log_key}gop_distortion_loss",
            logging_metrics.gop_distortion_loss / reduction,
        )
        self.log(f"{log_key}gop_bpp", logging_metrics.gop_bpp / reduction)
        if logging_metrics.gop_flow_bpp is not None:
            self.log(f"{log_key}gop_flow_bpp", logging_metrics.gop_flow_bpp / reduction)
        if logging_metrics.gop_residual_bpp is not None:
            self.log(
                f"{log_key}gop_residual_bpp",
                logging_metrics.gop_residual_bpp / reduction,
            )

        # logging metrics
        self.psnr(torch.cat(image2_list), torch.cat(image2_est_list))
        self.log(f"{log_key}psnr", self.psnr)

    def training_step(self, batch, batch_idx):
        self.update(force=True)
        log_key = f"{self.training_stage}/train_"

        if isinstance(self.optimizers(), list):
            [opt1, opt2] = self.optimizers()
        else:
            opt1 = self.optimizers()
            opt2 = None

        # compress the iframe and get its bpp cost (no grads)
        # TODO:why do we do this?
        batch, iframe_bpp = self.compress_iframe(batch)

        # update main model params
        # gop = "Group of Pictures"
        time_start = time.time()
        logging_metrics = LoggingMetrics(
            gop_total_loss=0,
            gop_distortion_loss=0,
            gop_bpp=iframe_bpp,
            gop_flow_bpp=0,
            gop_residual_bpp=0,
        )
        image2_list = []
        image2_est_list = []

        image1 = batch[:, 0]


        if self.on_sphere_learning and (self.training_stage == "4_total_2frame" or self.training_stage == "5_total"):
            for i in range(self.num_pframes):
                opt1.zero_grad()  # we backprop for every P-frame
                image2 = batch[:, i + 1]
                # Number of patches taken from each sample each time
                list_patch_ids = self.generate_random_patches(
                    self.healpix_resolution_patch_level) if not self.noPatching else [0]
                # list_patch_ids = [170]
                # print(list_patch_ids)

                for patch_id in list_patch_ids:

                    opt1.zero_grad()  # we backprop for every P-frame

                    dict_index = dict()
                    dict_weight = dict()
                    for r in self.list_res:
                        if self.struct_loader.__class__.__name__ == "HealpixSdpaStructLoader":
                            if self.noPatching:
                                dict_index[r], dict_weight[r], _ = self.struct_loader.getStruct(sampling_res=r, num_hops=1,
                                                                                           patch_res=None,
                                                                                           patch_id=patch_id)
                            else:
                                dict_index[r], dict_weight[r], _, _, _ = self.struct_loader.getStruct(sampling_res=r[0],
                                                                                                 num_hops=1,
                                                                                                 patch_res=r[1],
                                                                                                 patch_id=patch_id)
                        # else:
                        #     dict_index[r], dict_weight[r], _, _ = self.struct_loader.getGraph(
                        #         sampling_res=r if self.noPatching else r[0],
                        #         patch_res=None if self.noPatching else r[1],
                        #         num_hops=0, patch_id=patch_id)

                    loss_values, images = self.model.compute_batch_loss(image1, image2, dict_index, dict_weight,
                                                                        self.sample_res, self.patch_res, patch_id,
                                                                        self.nPix_per_patch)
                    # loss function collection
                    loss, logging_metrics = self.compute_loss_and_metrics(
                        loss_values, logging_metrics
                    )
                    self.manual_backward(loss)
                    torch.nn.utils.clip_grad_norm_(
                        opt1.param_groups[0]["params"], self.grad_clip_value
                    )
                    opt1.step()

                image1 = images.image2_est  # images are detached
                # keep track of these for other distortion metrics
                # note: these have no grads
                image2_list.append(images.image2)
                image2_est_list.append(images.image2_est)
        else:
            for i in range(self.num_pframes):

                opt1.zero_grad()  # we backprop for every P-frame
                image2 = batch[:, i + 1]
                loss_values, images = self.model.compute_batch_loss(image1, image2)

                image1 = images.image2_est  # images are detached

                # keep track of these for other distortion metrics
                # note: these have no grads
                image2_list.append(images.image2)
                image2_est_list.append(images.image2_est)

                # loss function collection

                loss, logging_metrics = self.compute_loss_and_metrics(
                    loss_values, logging_metrics
                )


                self.manual_backward(loss)
                torch.nn.utils.clip_grad_norm_(
                    opt1.param_groups[0]["params"], self.grad_clip_value
                )
                opt1.step()


        # lr step
        if self.lr_schedulers() is not None:
            self.lr_schedulers().step()

        # stat reductions and logging
        reduction = self.num_pframes + 1
        self.log_all_metrics(
            log_key, reduction, image2_list, image2_est_list, logging_metrics
        )

        # auxiliary update
        # this is the loss for learning the quantiles of the bottlenecks.
        if opt2 is not None:
            opt2.zero_grad()
            aux_loss = self.model.quantile_loss()
            self.log(f"{log_key}quantile_loss", aux_loss, sync_dist=True)
            self.manual_backward(aux_loss)
            opt2.step()

    def on_train_end(self) -> None:
        self.update(force=True)

    def validation_step(self, batch, batch_idx):
        self.update(force=True)
        log_key = f"{self.training_stage}/val_"
        # gop = "Group of Pictures"
        logging_metrics = LoggingMetrics(
            gop_total_loss=0,
            gop_distortion_loss=0,
            gop_bpp=0,
            gop_flow_bpp=0,
            gop_residual_bpp=0,
        )
        image2_list = []
        image2_est_list = []

        batch, gop_bpp = self.compress_iframe(batch)  # bpp_total w/o grads
        image1 = batch[:, 0]
        if self.on_sphere_learning and (self.training_stage == "4_total_2frame" or self.training_stage == "5_total"):
            for i in range(self.num_pframes):
                image2 = batch[:, i + 1]

                # Number of patches taken from each sample each time
                list_patch_ids = self.generate_random_patches(self.healpix_resolution_patch_level,
                                                              ) if not self.noPatching else [0]

                for patch_id in list_patch_ids:

                    dict_index = dict()
                    dict_weight = dict()
                    for r in self.list_res:
                        if self.struct_loader.__class__.__name__ == "HealpixSdpaStructLoader":
                            if self.noPatching:
                                dict_index[r], dict_weight[r], _ = self.struct_loader.getStruct(sampling_res=r, num_hops=1,
                                                                                           patch_res=None,
                                                                                           patch_id=patch_id)
                            else:
                                dict_index[r], dict_weight[r], _, _, _ = self.struct_loader.getStruct(sampling_res=r[0],
                                                                                                 num_hops=1,
                                                                                                 patch_res=r[1],
                                                                                                 patch_id=patch_id)
                        # else:
                        #     dict_index[r], dict_weight[r], _, _ = self.struct_loader.getGraph(
                        #         sampling_res=r if self.noPatching else r[0],
                        #         patch_res=None if self.noPatching else r[1],
                        #         num_hops=0, patch_id=patch_id)

                    loss_values, images = self.model.compute_batch_loss(image1, image2, dict_index, dict_weight,
                                                                        self.sample_res, self.patch_res, patch_id,
                                                                        self.nPix_per_patch)
                    # loss function collection
                    loss, logging_metrics = self.compute_loss_and_metrics(
                        loss_values, logging_metrics
                    )

                image1 = images.image2_est  # images are detached
                # keep track of these for other distortion metrics
                image2_list.append(images.image2)
                image2_est_list.append(images.image2_est)
        else:
            for i in range(self.num_pframes):
                image2 = batch[:, i + 1]
                loss_values, images = self.model.compute_batch_loss(image1, image2)
                image1 = images.image2_est  # images are detached

                # keep track of these for other distortion metrics
                image2_list.append(images.image2)
                image2_est_list.append(images.image2_est)

                # loss function collection
                loss, logging_metrics = self.compute_loss_and_metrics(
                    loss_values, logging_metrics
                )

        # stat reductions and logging
        reduction = self.num_pframes + 1
        self.log("val_loss", loss)
        self.log_all_metrics(
            log_key, reduction, image2_list, image2_est_list, logging_metrics
        )

    # TODO: implement test_step
    def on_test_start(self):
        print("Starting the Test Loop.")
        self.update(force=True)

    def test_step(self, batch, batch_idx):
        # gop = "Group of Pictures". gop_bpp is for i-frame compression
        # TODO: consider add bpp for i-frame compression to gop_bpp at start
        logging_metrics = LoggingMetrics(
            gop_total_loss=0,
            gop_distortion_loss=0,
            gop_bpp=0,
            gop_flow_bpp=0,
            gop_residual_bpp=0,
        )
        image2_list = []
        image2_est_list = []

        batch, gop_bpp = self.compress_iframe(batch)  # bpp_total w/o grads
        image1 = batch[:, 0]
        if self.on_sphere_learning:
            for i in range(self.num_pframes):
                image2 = batch[:, i + 1]

                # Number of patches taken from each sample each time
                list_patch_ids = self.generate_random_patches(self.healpix_resolution_patch_level,
                                                              ) if not self.noPatching else [0]
                # list_patch_ids = [170]
                # print(list_patch_ids)

                for patch_id in list_patch_ids:

                    dict_index = dict()
                    dict_weight = dict()
                    for r in self.list_res:
                        if self.struct_loader.__class__.__name__ == "HealpixSdpaStructLoader":
                            if self.noPatching:
                                dict_index[r], dict_weight[r], _ = self.struct_loader.getStruct(sampling_res=r, num_hops=1,
                                                                                           patch_res=None,
                                                                                           patch_id=patch_id)
                            else:
                                dict_index[r], dict_weight[r], _, _, _ = self.struct_loader.getStruct(sampling_res=r[0],
                                                                                                 num_hops=1,
                                                                                                 patch_res=r[1],
                                                                                                 patch_id=patch_id)

                    loss_values, images = self.model.compute_batch_loss(image1, image2, dict_index, dict_weight,
                                                                        self.sample_res, self.patch_res, patch_id,
                                                                        self.nPix_per_patch)
                    # loss function collection
                    loss, logging_metrics = self.compute_loss_and_metrics(
                        loss_values, logging_metrics
                    )

                image1 = images.image2_est  # images are detached
                # keep track of these for other distortion metrics
                image2_list.append(images.image2)
                image2_est_list.append(images.image2_est)
        else:
            for i in range(self.num_pframes):
                image2 = batch[:, i + 1]
                loss_values, images = self.model.compute_batch_loss(image1, image2)
                image1 = images.image2_est  # images are detached

                # keep track of these for other distortion metrics
                image2_list.append(images.image2)
                image2_est_list.append(images.image2_est)

                # loss function collection
                loss, logging_metrics = self.compute_loss_and_metrics(
                    loss_values, logging_metrics
                )

        # stat reductions and logging
        reduction = self.num_pframes + 1

        bpp = logging_metrics.gop_bpp / reduction
        print(self.psnr(torch.cat(image2_list), torch.cat(image2_est_list)))
        print(bpp)
        print(ms_ssim(torch.cat(image2_list), torch.cat(image2_est_list), data_range=1.0))
        return {'bpp': bpp, 'psnr': self.psnr(torch.cat(image2_list), torch.cat(image2_est_list)), 'ms_ssim': ms_ssim(torch.cat(image2_list), torch.cat(image2_est_list), data_range=1.0)}


    def compress_and_decompress(self, batch):
        self.update(force=True)


        image2_list = []
        image2_est_list = []
        image1_ori = batch[:, 0].permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy()
        batch, gop_bpp = self.compress_iframe(batch)  # bpp_total w/o grads
        image1 = batch[:, 0]
        image1_reconstructed = image1.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy()
        if self.on_sphere_learning and (self.training_stage == "4_total_2frame" or self.training_stage == "5_total"):
            for i in range(self.num_pframes):
                image2 = batch[:, i + 1]

                # Number of patches taken from each sample each time
                list_patch_ids = self.generate_random_patches(self.healpix_resolution_patch_level,
                                                              ) if not self.noPatching else [0]

                for patch_id in list_patch_ids:

                    dict_index = dict()
                    dict_weight = dict()
                    for r in self.list_res:
                        if self.struct_loader.__class__.__name__ == "HealpixSdpaStructLoader":
                            if self.noPatching:
                                dict_index[r], dict_weight[r], _ = self.struct_loader.getStruct(sampling_res=r, num_hops=1,
                                                                                           patch_res=None,
                                                                                           patch_id=patch_id)
                            else:
                                dict_index[r], dict_weight[r], _, _, _ = self.struct_loader.getStruct(sampling_res=r[0],
                                                                                                 num_hops=1,
                                                                                                 patch_res=r[1],
                                                                                                 patch_id=patch_id)
                        # else:
                        #     dict_index[r], dict_weight[r], _, _ = self.struct_loader.getGraph(
                        #         sampling_res=r if self.noPatching else r[0],
                        #         patch_res=None if self.noPatching else r[1],
                        #         num_hops=0, patch_id=patch_id)

                    loss_values, output = self.model.compute_batch_loss(image1, image2, dict_index, dict_weight,
                                                                        self.sample_res, self.patch_res, patch_id,
                                                                        self.nPix_per_patch)


                image1 = output.image2_est  # images are detached
                # keep track of these for other distortion metrics
                image2_list.append(image2.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
                image2_est_list.append(output.image2_est.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())


        else:
            for i in range(self.num_pframes):
                image2 = batch[:, i + 1]
                loss, output = self.model.compute_batch_loss(image1, image2)
                image1 = output.image2_est  # images are detached

                # keep track of these for other distortion metrics
                image2_list.append(image2.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())
                image2_est_list.append(output.image2_est.permute(0, 2, 3, 1).squeeze(dim=0).detach().numpy())


        # # Determine the number of images in each sequence
        # num_images2 = len(image2_list) + 1
        # num_images2_est = len(image2_est_list) + 1
        # #
        # # # Compute the figure size
        # fig_height_per_subplot = 256 / 300  # assuming 80 dpi, you can adjust this
        # fig_width_per_subplot = 256 / 300  # assuming 80 dpi, you can adjust this
        # fig_width = (max(num_images2, num_images2_est) + 1) * fig_width_per_subplot
        # fig_height = 2 * fig_height_per_subplot  # 3 rows
        #
        # # Create a new figure
        # fig, axes = plt.subplots(2, num_images2, figsize=(fig_width, fig_height))
        #
        # # Remove extra space between subplots
        # plt.subplots_adjust(hspace=0.05, wspace=0.05)  # No extra space
        # for i in range(num_images2):
        #     if i == 0:
        #         axes[0, i].imshow(image1_ori)
        #     axes[0, i].imshow(image2_list[i - 1])
        #     # axes[0, i].set_title(f"Image2_{i + 1}")
        #     axes[0, i].axis('off')
        #
        # # Plot third sequence of images in the third row
        # for i in range(num_images2_est):
        #     if i == 0:
        #         axes[1, i].imshow(image1_reconstructed)
        #     axes[1, i].imshow(image2_est_list[i - 1])
        #     # axes[1, i].set_title(f"Image2Est_{i + 1}")
        #     axes[1, i].axis('off')
        # prefix = 'performance'
        # path = prefix + '/compare' + '.png'
        # if not os.path.exists(prefix):
        #     os.makedirs(prefix)
        # plt.savefig('performance/compare.png', dpi=300)
        # plt.clf()

        #
        # to_be_compressed = image2_list[2]
        # to_be_compressed = (to_be_compressed * 255).astype(np.uint8)
        # cv2.imwrite("performance/to_be_compressed.png", cv2.cvtColor(to_be_compressed, cv2.COLOR_RGB2BGR))
        #
        # image1_to_be_saved = (image1_to_be_saved * 255).astype(np.uint8)
        # cv2.imwrite("performance/image2_est.png", cv2.cvtColor(image1_to_be_saved, cv2.COLOR_RGB2BGR))
        # for i in range(2):
        #     image2_to_be_saved = image2_est_list[i]
        #     image2_to_be_saved = (image2_to_be_saved * 255).astype(np.uint8)
        #     cv2.imwrite("performance/image2_est" + str(i) + ".png", cv2.cvtColor(image2_to_be_saved, cv2.COLOR_RGB2BGR))



    # def compress
    def test_epoch_end(self, outputs):
        avg_bpp = torch.stack([x['bpp'] for x in outputs])
        avg_psnr = torch.stack([x['psnr'] for x in outputs])
        prefix = 'performance'
        LineWidth = 2
        test, = plt.plot(avg_bpp, avg_psnr, marker='x', color='black', linewidth=LineWidth, label='new')

        # bpp, psnr, msssim = [0.176552, 0.107806, 0.074686, 0.052697], [37.754576, 36.680327, 35.602740, 34.276196], [
        #     0.970477, 0.963935, 0.955738, 0.942226]
        # baseline, = plt.plot(bpp, psnr, "b-*", linewidth=LineWidth, label='baseline')
        #
        # # Ours very fast
        # bpp, psnr, msssim = [0.187701631, 0.122491399, 0.084205003, 0.046558501], [36.52492847, 35.78201761,
        #                                                                            35.05371763, 33.56996097], [
        #     0.968154218, 0.962246563, 0.956369263, 0.942897242]
        # h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')
        #
        # bpp, psnr = [0.165663191, 0.109789007, 0.074090183, 0.039677747], [37.29259129, 36.5842637, 35.88754734,
        #                                                                    34.46536633]
        # h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')

        savepathpsnr = prefix + '/UVG_psnr' + '.png'
        print(prefix)
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        # plt.legend(handles=[h264, h265, baseline, test], loc=4)
        plt.legend(handles=[test], loc=4)
        plt.grid()
        plt.xlabel('Bpp')
        plt.ylabel('PSNR')
        plt.title('UVG dataset')
        plt.savefig(savepathpsnr)
        plt.clf()

    def configure_optimizers(self):
        # we have to train the model and the entropy bottleneck quantiles
        # separately
        model_param_dict, quantile_param_dict = self.model.collect_parameters()

        base_optim = optim.Adam(model_param_dict.values(), lr=self.learning_rate)
        if self.lr_scheduler_params is not None:
            scheduler = optim.lr_scheduler.StepLR(
                base_optim, **self.lr_scheduler_params
            )
            optimizers = [
                {
                    "optimizer": base_optim,
                    "lr_scheduler": scheduler,
                }
            ]
        else:
            optimizers = [{"optimizer": base_optim}]

        if quantile_param_dict is not None:
            optimizers.append(
                {
                    "optimizer": optim.Adam(
                        quantile_param_dict.values(),
                        lr=self.aux_learning_rate,
                    )
                }
            )

        return optimizers
