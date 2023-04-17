import logging
import os

import hydra
import pytorch_lightning as pl
import torchvision.transforms as T
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from models.callbacks import SampleImages

# from models.vae import VAE
from models.vae import VAE
from util import configure_device, get_dataset

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs")
def train(config):
    # Get config and setup
    config = config.dataset.vae
    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config.training.seed, workers=True)

    # Dataset
    root = config.data.root
    d_type = config.data.name
    image_size = config.data.image_size
    dataset = get_dataset(d_type, root, image_size, norm=False, flip=config.data.hflip)
    N = len(dataset)
    batch_size = config.training.batch_size
    batch_size = min(N, batch_size)

    # Model
    vae = VAE(
        input_res=image_size,
        enc_block_str=config.model.enc_block_config,
        dec_block_str=config.model.dec_block_config,
        enc_channel_str=config.model.enc_channel_config,
        dec_channel_str=config.model.dec_channel_config,
        lr=config.training.lr,
        alpha=config.training.alpha,
        contrastive=config.training.contrastive,
        batch_size=batch_size,
        c_weight=config.training.c_weight,
        max_c_weight=config.training.max_c_weight,
        decay_c_rate=config.training.decay_c_rate,
        masking_prob=config.training.masking_prob,
        square=config.training.square,
        pid=config.training.pid,
        dpi=config.training.dpi,
        c_max=config.training.c_max,
    )

    # Trainer
    train_kwargs = {}
    restore_path = config.training.restore_path
    if restore_path is not None:
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    results_dir = config.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"vae-{config.training.chkpt_prefix}"
        + "-{epoch:02d}-{train_loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
        save_on_train_epoch_end=True,
    )

    image_callback = SampleImages(log_dir=os.path.join(results_dir, "images"))

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs
    train_kwargs["log_every_n_steps"] = config.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback, image_callback]

    loader_kws = {}

    # Half precision training
    if config.training.fp16:
        train_kwargs["precision"] = 16

    # Loader
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(vae, train_dataloaders=loader)


if __name__ == "__main__":
    train()
