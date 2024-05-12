import wandb
import torch
import wandb.sdk

from pathlib import Path
from pytorch_lightning import LightningModule
from typing import Optional

from .callbacks import S3SyncCallback


def setup_s3_model_checkpointing(
    project_name: str, 
    wandb_run: Optional[wandb.sdk.wandb_run.Run]=None,
    load_from_run: Optional[str]=None,
    every_n_epochs: int=100,
) -> S3SyncCallback:
    """
    Set up the paths for model checkpointing and necessary s3 syncing.
    IDs are set using wandb runs

    Args:
    - project_name : name of project, needs to be consistent for s3 folder structure and wandb logging
    - wandb_run: the instance of the wandb run, if any
    - load_from_run: set this value to download the checkpoint of this wandb run
    """
    # setting up checkpoint 
    save_ckpt_local_dir = Path(
        f"./{project_name}/{wandb_run.id if wandb_run else 'local_run'}/checkpoints"
    )
    save_ckpt_local_dir.mkdir(parents=True, exist_ok=True)

    load_ckpt_local_dir = None
    if load_from_run:
        load_ckpt_local_dir = Path(f"./{project_name}/{load_from_run}/checkpoints")
        load_ckpt_local_dir.mkdir(parents=True, exist_ok=True)

    s3_sync_callback = S3SyncCallback(
        save_local_dir=save_ckpt_local_dir, 
        load_local_dir=load_ckpt_local_dir, 
        every_n_epochs=every_n_epochs,
    )

    if load_ckpt_local_dir is not None:
        s3_sync_callback.download_files_from_s3()

    return s3_sync_callback


def set_model_weight_from_checkpoint(
    model: LightningModule,
    s3_sync_callback: S3SyncCallback,
    resume_filename: str="last.ckpt",
) -> LightningModule:
    assert s3_sync_callback.load_local_dir is not None, "No loading directory specified in s3 sync."
    assert (s3_sync_callback.load_local_dir / resume_filename).exists()

    state_dict = torch.load((s3_sync_callback.load_local_dir / resume_filename))
    model.load_state_dict(state_dict["state_dict"])
    return model
