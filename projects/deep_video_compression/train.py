# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys

import hydra
import numpy as np
import os
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_module import VRHM48VideoLightning
from dvc_module import DvcModule
from neuralcompression.functional import optical_flow_to_color
from neuralcompression.models import DVC
# from neuralcompression.models.deep_video_compression import DVCResidualEncoder, DVCResidualDecoder
from projects.deep_video_compression.preprocessing import healpix_sdpa_struct_loader

sys.path.append("../..")

from spherical_models import SphereFactorizedPrior


class WandbImageCallback(pl.Callback):
    """
    Logs the input and output images of a module.

    Images are stacked into a mosaic, with different clips from top to bottom
    and time progressing from left to right.

    Args:
        batch: A set of images to log at the end of each validation epoch.
    """

    def __init__(self, batch):
        super().__init__()
        self.batch = batch

    def append_images(self, image_dict, image_group):
        keys = ("flow", "image1", "image2", "image2_est")
        for i, key in enumerate(keys):
            if image_group[i] is not None:
                image_dict.setdefault(key, []).append(image_group[i])
        return image_dict

    def log_images(self, trainer, base_key, image_dict, global_step):
        # check if we have images to log
        # if we do, then concatenate time along x-axis and batch along y-axis
        # and write
        keys = ("flow", "image1", "image2", "image2_est")
        for key in keys:
            if image_dict.get(key) is not None:
                caption = f"{key} (y-axis: batch, x-axis: time)"
                mosaic = torch.cat(image_dict[key], dim=-1)
                mosaic = torch.cat(list(mosaic), dim=-2)
                if key == "flow":
                    mosaic = optical_flow_to_color(mosaic.unsqueeze(0))[0]
                # TODO: find out why there is a clip operation here
                mosaic = torch.clip(mosaic, min=0, max=1.0)
                trainer.logger.experiment.log(
                    {
                        f"{base_key}/{key}": wandb.Image(mosaic, caption=caption),
                        "global_step": global_step,
                    }
                )
    # TODO: validation 不加噪声
    def on_validation_end(self, trainer, pl_module):
        image_dict = {}
        batch = self.batch.to(device=pl_module.device, dtype=pl_module.dtype)
        batch, _ = pl_module.compress_iframe(batch)  # bpp_total w/o grads
        image1 = batch[:, 0]
        if pl_module.on_sphere_learning and (pl_module.training_stage == "4_total_2frame" or pl_module.training_stage == "5_total"):
            for i in range(pl_module.num_pframes):
                image2 = batch[:, i + 1]

                # Number of patches taken from each sample each time
                list_patch_ids = pl_module.generate_random_patches(pl_module.healpix_resolution_patch_level,
                                                                   ) if not pl_module.noPatching else [0]
                # list_patch_ids = [170]
                # print(list_patch_ids)

                for patch_id in list_patch_ids:

                    dict_index = dict()
                    dict_weight = dict()
                    for r in pl_module.list_res:
                        if pl_module.struct_loader.__class__.__name__ == "HealpixSdpaStructLoader":
                            if pl_module.noPatching:
                                dict_index[r], dict_weight[r], _ = pl_module.struct_loader.getStruct(sampling_res=r, num_hops=1,
                                                                                           patch_res=None,
                                                                                           patch_id=patch_id)
                            else:
                                dict_index[r], dict_weight[r], _, _, _ = pl_module.struct_loader.getStruct(sampling_res=r[0],
                                                                                                 num_hops=1,
                                                                                                 patch_res=r[1],
                                                                                                 patch_id=patch_id)
                        # else:
                        #     dict_index[r], dict_weight[r], _, _ = self.struct_loader.getGraph(
                        #         sampling_res=r if self.noPatching else r[0],
                        #         patch_res=None if self.noPatching else r[1],
                        #         num_hops=0, patch_id=patch_id)

                    _, images = pl_module.model.compute_batch_loss(image1, image2, dict_index, dict_weight,
                                                                   pl_module.sample_res, pl_module.patch_res, patch_id,
                                                                   pl_module.nPix_per_patch)

                # _, images = pl_module.model.compute_batch_loss(image1, image2)

                image1 = images.image2_est  # images are detached

                image_dict = self.append_images(image_dict, images)
        else:
            for i in range(pl_module.num_pframes):
                image2 = batch[:, i + 1]
                _, images = pl_module.model.compute_batch_loss(image1, image2)
                image1 = images.image2_est  # images are detached

                image_dict = self.append_images(image_dict, images)

        self.log_images(
            trainer,
            f"log_images_stage_{pl_module.training_stage}",
            image_dict,
            pl_module.global_step,
        )


def merge_configs(cfg1, cfg2):
    """Handy config merger based on dictionaries."""
    new_cfg = cfg1.copy()
    OmegaConf.set_struct(new_cfg, False)
    new_cfg.update(cfg2)

    return new_cfg


def run_training_stage(stage, root, model, data, logger, image_logger, cfg):
    """Run a single training stage based on the stage config."""
    print(f"training stage: {stage}")
    stage_cfg = cfg.training_stages[stage]
    if stage_cfg.save_dir is None:
        save_dir = root / stage
    else:
        save_dir = Path(stage_cfg.save_dir)

    if (
        not cfg.checkpoint.overwrite
        and not cfg.checkpoint.resume_training
        and len(list(save_dir.glob("*.ckpt"))) > 0
    ):
        raise RuntimeError(
            "Checkpoints detected in save directory: set resume_training=True "
            "to restore trainer state from these checkpoints, "
            "or set overwrite=True to ignore them."
        )

    save_dir.mkdir(exist_ok=True, parents=True)
    last_checkpoint = save_dir / "last.ckpt"

    if not last_checkpoint.exists() or cfg.checkpoint.overwrite is True:
        last_checkpoint = None

    print("Last Checkpoint:", last_checkpoint)

    struct_loader = healpix_sdpa_struct_loader.HealpixSdpaStructLoader(weight_type='identity',
                                                                       use_geodesic=True,
                                                                       use_4connectivity=False,
                                                                       normalization_method='non',
                                                                       cutGraphForPatchOutside=True,
                                                                       load_save_folder='/scratch/zczqyc4/neighbor_structure')

    lightning_model = DvcModule(model, struct_loader, **merge_configs(cfg.module, stage_cfg.module), patching=False)

    trainer = pl.Trainer(
        **merge_configs(cfg.trainer, stage_cfg.trainer),
        logger=logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(dirpath=save_dir, **cfg.checkpoint.model_checkpoint),
            image_logger,
        ],
        resume_from_checkpoint=last_checkpoint,
    )

    trainer.fit(lightning_model, datamodule=data)

    return lightning_model.recompose_model(model)


@hydra.main(config_path="config", config_name="base")
def main(cfg: DictConfig):
    root = Path(cfg.logging.save_root)  # if relative, uses Hydra outputs dir
    model = DVC(**cfg.model)
    # model = DVC(**cfg.model,residual_network=SphereFactorizedPrior(64, 128))
    logger = WandbLogger(
        save_dir=str(root.absolute()),
        # project="DVC",
        project="DVC_VRHM48",
        # project="DVC_Spherical_no_patching_lambda=2048_more_steps_test",
        # project="DVC_Spherical_patching",
        config=OmegaConf.to_container(cfg),  # saves the Hydra config to wandb
        id="k916hof7" # specify run_id of a wandb project to resume training
    )
    # data = Vimeo90kSeptupletLightning(
    #     frames_per_group=7,
    #     **cfg.data,
    #     pin_memory=cfg.ngpu != 0,
    # )
    data = VRHM48VideoLightning(
        frames_per_group=7,
        **cfg.data,
        pin_memory=cfg.ngpu != 0,
    )

    # set up image logging
    rng = np.random.default_rng(cfg.logging.image_seed)
    data.setup()
    val_dataset = data.val_dataset
    log_image_indices = rng.permutation(len(val_dataset))[: cfg.logging.num_log_images]
    log_images = torch.stack([val_dataset[ind] for ind in log_image_indices])
    image_logger = WandbImageCallback(log_images)

    # run through each stage and optimize
    for stage in sorted(cfg.training_stages.keys()):
        model = run_training_stage(stage, root, model, data, logger, image_logger, cfg)

    # stage = sorted(cfg.training_stages.keys())[0]
    # model = run_training_stage(stage, root, model, data, logger, image_logger, cfg)
    #
    # stage = sorted(cfg.training_stages.keys())[1]
    # model = run_training_stage(stage, root, model, data, logger, image_logger, cfg)
    #
    # stage = sorted(cfg.training_stages.keys())[2]
    # model = run_training_stage(stage, root, model, data, logger, image_logger, cfg)
    #
    # stage = sorted(cfg.training_stages.keys())[3]
    # model = run_training_stage(stage, root, model, data, logger, image_logger, cfg)



if __name__ == "__main__":
    main()
