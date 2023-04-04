import torch
from gencon.utils import spherical_interpolation
from models.vae import VAE
from util import get_dataset
import yaml
import pandas as pd


def interpolate(mu1, mu2, x1, x2, pl_module, num_samples, spherical):
    shape = list(mu1.shape)
    shape[0] = num_samples
    batch_size = mu1.shape[0]

    imgs = []

    if spherical:
        sq_dims = 0
        while mu1.shape[-1] == 1:
            mu1 = torch.squeeze(mu1, -1)
            mu2 = torch.squeeze(mu2, -1)
            sq_dims += 1
        batchwise_interpol = spherical_interpolation(mu1, mu2, num_samples + 2)

    for j in range(min(num_samples, batch_size)):
        if not spherical:
            z = torch.zeros(shape)
            for i in range(0, shape[1]):
                z[:, i, 0, 0] = torch.linspace(
                    mu1[j, i, 0, 0].detach(), mu2[j, i, 0, 0].detach(), num_samples
                )
        else:
            z = batchwise_interpol[j]
            for i in range(sq_dims):
                z = torch.unsqueeze(z, -1)
        if x1 is not None:
            imgs.append(torch.unsqueeze(x1[j], 0))
        imgs.append(pl_module.decode(z.cuda()))
        if x2 is not None:
            imgs.append(torch.unsqueeze(x2[j], 0))
    return torch.cat(imgs, 0)


def interpolate_two_images(pl_module, x1, x2, num_samples, spherical=False):
    mu1, _ = pl_module.encode(x1)
    mu2, _ = pl_module.encode(x2)
    return interpolate(mu1, mu2, x1, x2, pl_module, num_samples, spherical)


def get_attr_indices(attribute_name, n_samples):
    lines = load_celeba_attributes()

    attribute_names = lines[1].split()
    attribute_index = attribute_names.index(attribute_name)

    attributes = []

    for line in lines[2:]:
        values = line.split()
        img_name = values[0]
        attr_value = int(values[attribute_index + 1])
        attributes.append((img_name, attr_value))

    attributes_df = pd.DataFrame(attributes, columns=["filename", attribute_name])

    with_attribute = attributes_df[attributes_df[attribute_name] == 1]
    without_attribute = attributes_df[attributes_df[attribute_name] == -1]

    if n_samples != -1:
        with_attribute = with_attribute.sample(n_samples)
        without_attribute = without_attribute.sample(n_samples)

    return with_attribute, without_attribute


def calculate_attribute_vectors(
    model, dataset, with_attribute, without_attribute, batch_size
):
    output = []

    for targets in [with_attribute, without_attribute]:
        latents = []
        imgs = []
        for idx in targets.index:
            if len(imgs) < batch_size:
                imgs.append(dataset[idx])
            if len(imgs) == batch_size:
                imgs = torch.stack(imgs).to("cuda")
                latent = model.encode(imgs)[0].detach().cpu()
                latents.append(latent)
                imgs = []
        output.append(torch.cat(latents, 0))

    return output


class Config(dict):
    def __getattr__(self, attr):
        value = self.get(attr, None)
        if isinstance(value, dict):
            return Config(value)
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_vae_model(chkp_path, device="cuda"):
    config = load_config()
    vae = VAE(
        input_res=config.data.image_size,
        enc_block_str=config.model.enc_block_config,
        dec_block_str=config.model.dec_block_config,
        enc_channel_str=config.model.enc_channel_config,
        dec_channel_str=config.model.dec_channel_config,
        batch_size=config.evaluation.batch_size,
    )
    ckpt = torch.load(chkp_path, map_location=torch.device(device))
    vae.load_state_dict(ckpt["state_dict"])
    return vae.eval().to("cuda")


def load_dataset():
    config = load_config()
    d_type = config.data.name
    image_size = config.data.image_size
    dataset = get_dataset(
        d_type, config.data.root, image_size, norm=False, flip=config.data.hflip
    )
    return dataset


def load_celeba_attributes():
    config = load_config()
    attributes_file_path = config.data.root + "/CelebAMask-HQ-attribute-anno.txt"

    with open(attributes_file_path, "r") as f:
        lines = f.readlines()

    return lines


def load_ddpm_model():
    # TODO: Implement this.
    return None


def load_config(
    config_path="main/configs/dataset/celebamaskhq128/test.yaml",
):
    with open(config_path) as cfg:
        config = Config(yaml.safe_load(cfg)["vae"])
    return config
