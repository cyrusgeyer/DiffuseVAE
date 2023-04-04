import os
from typing import Sequence, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor
from torch.nn import Module
from util import save_as_images, save_as_np
from util_gencon import interpolate_two_images
import torchvision.transforms as ttf
import torchvision.utils as vutils


class SampleImages(Callback):
    def __init__(self, log_dir, num_samples=24) -> None:
        super().__init__()
        self.log_dir = log_dir
        self.num_samples = num_samples
        self.frq = 200
        self.last_step = -self.frq

        self.t = torch.jit.script(
            torch.nn.Sequential(
                ttf.RandomHorizontalFlip(p=1.0),
            )
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not trainer.is_global_zero:
            return
        if trainer.global_step - self.last_step < self.frq:
            return
        self.last_step = trainer.global_step

        x1 = pl_module.last_batch
        x2 = self.t(x1)
        imgs = interpolate_two_images(pl_module, x1, x2, self.num_samples)

        dir_path = os.path.join(self.log_dir, "Samples")
        os.makedirs(dir_path, exist_ok=True)
        vutils.save_image(
            imgs.cpu().data,
            os.path.join(
                dir_path,
                f"Epoch_{pl_module.current_epoch}_Step_{pl_module.global_step}.png",
            ),
            normalize=True,
            nrow=self.num_samples + 2,
        )


class EMAWeightUpdate(Callback):
    """EMA weight update
    Your model should have:
        - ``self.online_network``
        - ``self.target_network``
    Updates the target_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.
    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    Example::
        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...
        trainer = Trainer(callbacks=[EMAWeightUpdate()])
    """

    def __init__(self, tau: float = 0.9999):
        """
        Args:
            tau: EMA decay rate
        """
        super().__init__()
        self.tau = tau

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # get networks
        online_net = pl_module.online_network.decoder
        target_net = pl_module.target_network.decoder

        # update weights
        self.update_weights(online_net, target_net)

    def update_weights(
        self, online_net: Union[Module, Tensor], target_net: Union[Module, Tensor]
    ) -> None:
        # apply MA weight update
        with torch.no_grad():
            for targ, src in zip(target_net.parameters(), online_net.parameters()):
                targ.mul_(self.tau).add_(src, alpha=1 - self.tau)


class ImageWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        write_interval,
        compare=False,
        n_steps=None,
        eval_mode="sample",
        conditional=True,
        sample_prefix="",
        save_vae=False,
        save_mode="image",
        is_norm=True,
    ):
        super().__init__(write_interval)
        assert eval_mode in ["sample", "recons"]
        self.output_dir = output_dir
        self.compare = compare
        self.n_steps = 1000 if n_steps is None else n_steps
        self.eval_mode = eval_mode
        self.conditional = conditional
        self.sample_prefix = sample_prefix
        self.save_vae = save_vae
        self.is_norm = is_norm
        self.save_fn = save_as_images if save_mode == "image" else save_as_np

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        rank = pl_module.global_rank
        if self.conditional:
            ddpm_samples_dict, vae_samples = prediction

            if self.save_vae:
                vae_samples = vae_samples.cpu()
                vae_save_path = os.path.join(self.output_dir, "vae")
                os.makedirs(vae_save_path, exist_ok=True)
                self.save_fn(
                    vae_samples,
                    file_name=os.path.join(
                        vae_save_path,
                        f"output_vae_{self.sample_prefix}_{rank}_{batch_idx}",
                    ),
                    denorm=self.is_norm,
                )
        else:
            ddpm_samples_dict = prediction

        # Write output images
        # NOTE: We need to use gpu rank during saving to prevent
        # processes from overwriting images
        for k, ddpm_samples in ddpm_samples_dict.items():
            ddpm_samples = ddpm_samples.cpu()

            # Setup dirs
            base_save_path = os.path.join(self.output_dir, k)
            img_save_path = os.path.join(base_save_path, "images")
            os.makedirs(img_save_path, exist_ok=True)

            # Save
            self.save_fn(
                ddpm_samples,
                file_name=os.path.join(
                    img_save_path, f"output_{self.sample_prefix }_{rank}_{batch_idx}"
                ),
                denorm=self.is_norm,
            )

        # FIXME: This is currently broken. Separate this from the core logic
        # into a new function. Uncomment when ready!
        # if self.compare:
        #     # Save comparisons
        #     (_, img_samples), _ = batch
        #     img_samples = normalize(img_samples).cpu()
        #     iter_ = vae_samples if self.eval_mode == "sample" else img_samples
        #     for idx, (ddpm_pred, pred) in enumerate(zip(ddpm_samples, iter_)):
        #         samples = {
        #             "VAE" if self.eval_mode == "sample" else "Original": pred,
        #             "DDPM": ddpm_pred,
        #         }
        #         compare_samples(
        #             samples,
        #             save_path=os.path.join(
        #                 self.comp_save_path, f"compare_form1_{rank}_{idx}.png"
        #             ),
        #         )
