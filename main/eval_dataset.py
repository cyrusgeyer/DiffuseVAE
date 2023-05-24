import argparse
import pickle
import os

import torch
from util_gencon import load_vae_model, load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader


def eval_dataset(model, dataset):
    mus = []
    log_vars = []

    for i, batch in tqdm(enumerate(dataset)):
        mu, log_var = model.encode(batch.to("cuda"))
        mus.append(mu.detach().cpu())
        log_vars.append(log_var.detach().cpu())

    mus = torch.cat(mus, 0)
    log_vars = torch.cat(log_vars, 0)

    return mus, log_vars


def main(args):
    dataset = load_dataset()

    data_loader = DataLoader(
        dataset,
        args.batch_size,
        num_workers=9,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    model = load_vae_model(args.checkpoint_path, training_gpus=args.training_gpus)

    mus, log_vars = eval_dataset(model, data_loader)

    output_dir = os.path.dirname(args.checkpoint_path)
    out_mus = os.path.join(output_dir, f"mus.pkl")
    out_log_var = os.path.join(output_dir, f"log_vars.pkl")

    with open(out_mus, "wb") as f:
        pickle.dump(mus, f)

    with open(out_log_var, "wb") as f:
        pickle.dump(log_vars, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute mean encodings for specified attributes"
    )
    parser.add_argument(
        "--checkpoint_path",
        help="Path to the PyTorch Lightning checkpoint",
        default="/cluster/work/buhmann/icu_rs/geyerr/PODVAE/GenCon/submodules/DiffuseVAE/outputs/2023-04-17/16-46-41/vae_celebaHQ_contrastive_300/checkpoints/vae-cmhq128_alpha=1.0-epoch=358-train_loss=0.0000.ckpt",
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size",
        default=16,
    )
    parser.add_argument(
        "--training_gpus",
        help="Number of GPUs used during training.",
        default=4,
    )

    args = parser.parse_args()
    main(args)
