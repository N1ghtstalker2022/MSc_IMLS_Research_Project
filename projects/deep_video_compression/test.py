import os
import sys
import hydra
import pytorch_lightning as pl
import torch
from dvc_module import DvcModule
from data_module import VRHM48VideoLightning
from projects.deep_video_compression.preprocessing import healpix_sdpa_struct_loader
from omegaconf import DictConfig, OmegaConf
from neuralcompression.models import DVC
sys.path.append("../..")

from spherical_models import SphereFactorizedPrior


def merge_configs(cfg1, cfg2):
    """Handy config merger based on dictionaries."""
    new_cfg = cfg1.copy()
    OmegaConf.set_struct(new_cfg, False)
    new_cfg.update(cfg2)

    return new_cfg
@hydra.main(config_path="config", config_name="base")
def main(cfg: DictConfig):
    # Initialize the Trainer

    struct_loader = healpix_sdpa_struct_loader.HealpixSdpaStructLoader(weight_type='identity',
                                                                       use_geodesic=True,
                                                                       use_4connectivity=False,
                                                                       normalization_method='non',
                                                                       cutGraphForPatchOutside=True,
                                                                       load_save_folder='/scratch/zczqyc4/neighbor_structure')
    stage_cfg = cfg.training_stages['s5_total']
    model = DVC(**cfg.model)
    # model = DVC(**cfg.model, residual_network=SphereFactorizedPrior(64, 128))
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    # Initialize your model from a checkpoint

    # last_checkpoint = "/home/zczqyc4/NeuralCompression/projects/deep_video_compression/outputs/2023-08-29/23-18-37/s3_motion_compensation/epoch=13-step=6999.ckpt"
    # last_checkpoint = "/home/zczqyc4/NeuralCompression/projects/deep_video_compression/outputs/2023-08-26/01-13-39/s5_total/last.ckpt" # DVC
    # last_checkpoint = "/home/zczqyc4/NeuralCompression/projects/deep_video_compression/outputs/2023-08-28/13-13-57/s5_total/epoch=10-step=5499.ckpt" # DVC_spherical_patching
    # last_checkpoint = "/home/zczqyc4/NeuralCompression/projects/deep_video_compression/outputs/2023-08-29/00-58-31/s5_total/last.ckpt" # DVC_spherical_no_patching
    last_checkpoint = "/home/zczqyc4/NeuralCompression/projects/deep_video_compression/outputs/2023-09-03/22-26-09/s5_total/last.ckpt" # DVC_VHRM48
    #

    # last_checkpoint = "/home/zczqyc4/NeuralCompression/projects/deep_video_compression/outputs/2023-09-02/14-49-20/s2_motion_compression/last.ckpt" # DVC_256
    # last_checkpoint = "/home/zczqyc4/NeuralCompression/projects/deep_video_compression/outputs/2023-09-02/14-43-39/s2_motion_compression/epoch=39-step=19999.ckpt" # DVC_512
    # last_checkpoint = "/home/zczqyc4/NeuralCompression/projects/deep_video_compression/outputs/2023-09-03/02-30-04/s2_motion_compression/last.ckpt" # DVC_1024
    # last_checkpoint = "/home/zczqyc4/NeuralCompression/projects/deep_video_compression/outputs/2023-09-03/13-40-26/s5_total/last.ckpt" # DVC_2048
    # last_checkpoint = "/home/zczqyc4/NeuralCompression/projects/deep_video_compression/outputs/2023-09-03/02-54-40/s5_total/last.ckpt" # DVC_2048

    if not os.path.exists(last_checkpoint):
        exit(-1)

    # lightning_model = DvcModule(model)
    # TODO: load the spherical model from checkpoint!
    checkpoint = torch.load(last_checkpoint)
    state_dict = checkpoint['state_dict']

    lightning_model = DvcModule.load_from_checkpoint(last_checkpoint, model=model, struct_loader=struct_loader, patching=False, **merge_configs(cfg.module, stage_cfg.module))

    # load test video data
    data = VRHM48VideoLightning(
        frames_per_group=7,
        **cfg.data,
        # pin_memory=cfg.ngpu != 0,
    )
    # data = Vimeo90kSeptupletLightning(
    #     frames_per_group=7,
    #     **cfg.data,
    #     # pin_memory=cfg.ngpu != 0,
    # )
    data.setup()

    trainer = pl.Trainer(
        gpus=0,
    )
    # Test the model TODO: (compress && decompress)
    # trainer.test(lightning_model, datamodule=data, ckpt_path=last_checkpoint)

    # test the video compression performance by PSNR and bpp, plot the rate-distortion curve (gop_distortion, gop_bpp)
    test_dataloader = data.test_dataloader()

    # Convert the DataLoader to an iterator and fetch the first batch
    test_iter = iter(test_dataloader)
    first_batch = next(test_iter)


    # plot the result of compression and decompression
    lightning_model.eval()
    lightning_model.compress_and_decompress(first_batch)

if __name__ == "__main__":
    main()