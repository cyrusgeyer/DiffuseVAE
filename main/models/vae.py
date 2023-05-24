def random_zero_square(tensor, dim=25):
    shape = tensor.shape[1:]

    mask = torch.ones(shape)
    xy = (torch.rand(2) * torch.tensor(shape)).int()

    x1 = xy[0]
    x2 = min(shape[0], xy[0] + dim)

    y1 = xy[1]
    y2 = min(shape[1], xy[1] + dim)

    mask[x1:x2, y1:y2] = 0
    return torch.stack([mask] * tensor.shape[0])


def random_zero_square_batch(batch, dim):
    masks = []
    for img in batch:
        masks.append(random_zero_square(img, dim))
    return torch.stack(masks)


def random_zero_mask(tensor, p):
    mask = torch.rand(tensor.shape[1:]) > p
    return torch.stack([mask] * tensor.shape[0])


def random_zero_mask_batch(batch, p=0.1):
    masks = []
    for img in batch:
        masks.append(random_zero_mask(img, p))
    return torch.stack(masks)


import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from gencon.sample import reparametrize_non_diag
from gencon.contrastive_loss import NT_Xent
from gencon.P_PID import PIDControlWrapper
from gencon.D_PI import IncrementalPIControllerWrapper
import torchvision.transforms as ttf
import numpy as np
import torchvision.transforms as ttf
import gpustat


def create_transforms(device):
    transforms_1 = torch.nn.Sequential(
        ttf.RandomAdjustSharpness(5.0, p=0.5),
        ttf.RandomHorizontalFlip(p=0.0),
    ).to(device)

    transforms_2 = torch.nn.Sequential(
        ttf.GaussianBlur(7, sigma=(0.1, 1.4)),
        ttf.RandomHorizontalFlip(p=1.0),
    ).to(device)

    scripted_transforms = [transforms_1, transforms_2]
    return scripted_transforms


def parse_layer_string(s):
    layers = []
    for ss in s.split(","):
        if "x" in ss:
            # Denotes a block repetition operation
            res, num = ss.split("x")
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif "u" in ss:
            # Denotes a resolution upsampling operation
            res, mixin = [int(a) for a in ss.split("u")]
            layers.append((res, mixin))
        elif "d" in ss:
            # Denotes a resolution downsampling operation
            res, down_rate = [int(a) for a in ss.split("d")]
            layers.append((res, down_rate))
        elif "t" in ss:
            # Denotes a resolution transition operation
            res1, res2 = [int(a) for a in ss.split("t")]
            layers.append(((res1, res2), None))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def parse_channel_string(s):
    channel_config = {}
    for ss in s.split(","):
        res, in_channels = ss.split(":")
        channel_config[int(res)] = int(in_channels)
    return channel_config


def get_conv(
    in_dim,
    out_dim,
    kernel_size,
    stride,
    padding,
    zero_bias=True,
    zero_weights=False,
    groups=1,
):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_width,
        middle_width,
        out_width,
        down_rate=None,
        residual=False,
        use_3x3=True,
        zero_last=False,
    ):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c3 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        ld = self.get_layer_devices()
        x.to(ld[0])
        xhat = self.c1(F.gelu(x))
        xhat.to(ld[1])
        xhat = self.c2(F.gelu(xhat))
        xhat = xhat.to(ld[2])
        xhat = self.c3(F.gelu(xhat))
        xhat.to(ld[3])
        xhat = self.c4(F.gelu(xhat))
        out = x.to(ld[3]) + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out

    def get_layer_devices(self):
        return [
            next(layer.parameters()).device
            for layer in [self.c1, self.c2, self.c3, self.c4]
        ]


class Encoder(nn.Module):
    def __init__(self, block_config_str, channel_config_str, device_list):
        super().__init__()
        self.device_list = device_list

        block_config = parse_layer_string(block_config_str)
        channel_config = parse_channel_string(channel_config_str)
        blocks = []
        for _, (res, down_rate) in enumerate(block_config):
            if isinstance(res, tuple):
                # Denotes transition to another resolution
                res1, res2 = res
                blocks.append(
                    nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)
                )
                continue
            in_channel = channel_config[res]
            use_3x3 = res > 1
            blocks.append(
                ResBlock(
                    in_channel,
                    int(0.5 * in_channel),
                    in_channel,
                    down_rate=down_rate,
                    residual=True,
                    use_3x3=use_3x3,
                )
            )
        # TODO: If the training is unstable try using scaling the weights
        self.in_conv = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        num_blocks = len(self.device_list)
        self.split_blocks = make_split_blocks(self, blocks, num_blocks, 0.55)
        # Latents
        self.mu = nn.Conv2d(channel_config[1], channel_config[1], 1, bias=False)

        self.logvar = nn.Conv2d(channel_config[1], channel_config[1], 1, bias=False)

    def forward(self, input):
        # Only executed at the very first forward pass
        self.move_layers_to_gpu()
        input = input.to(self.device_list[0])
        x = self.in_conv(input)
        x = distributed_forwarding(x, self.device_list, self.split_blocks)
        return self.mu(x), self.logvar(x)

    def move_layers_to_gpu(self):
        self.in_conv.to(self.device_list[0])
        for i, blocks in enumerate(self.split_blocks):
            blocks.to(self.device_list[i])
        self.mu.to(self.device_list[-1])
        self.logvar.to(self.device_list[-1])


def distributed_forwarding(x, device_list, split_blocks):
    for i, blocks in enumerate(split_blocks):
        x = blocks(x.to(device_list[i]))
    return x


def make_split_blocks(module, blocks, num_devices, n):
    split_blocks = []
    num_blocks = len(blocks)

    # Calculate the total memory usage assuming each layer is n times larger than the one before
    total_memory = sum((n**i) for i in range(num_blocks))

    # Calculate the target memory per device
    target_memory_per_device = total_memory / num_devices

    current_device_memory = 0
    current_device_idx = 0
    current_device_blocks = []
    for i in range(len(blocks)):
        if current_device_idx == num_devices - 1:
            break
        current_device_memory += n**i
        current_device_blocks.append(blocks[i])

        # If the current device memory is close to the target or it's the last device, create a new block
        if current_device_memory >= target_memory_per_device:
            block = nn.Sequential(*current_device_blocks)
            split_blocks.append(block)
            setattr(module, f"block_{current_device_idx}", block)
            current_device_idx += 1

            # Reset the current device memory and blocks for the next device
            current_device_memory = 0
            current_device_blocks = []

    print(f"LAST INDEX: {i}")
    print(f"Num Blocks: {num_blocks}")
    block = nn.Sequential(*blocks[i:])
    split_blocks.append(block)
    setattr(module, f"block_{current_device_idx}", block)

    assert len(split_blocks) == num_devices
    return split_blocks


class Decoder(nn.Module):
    def __init__(self, input_res, block_config_str, channel_config_str, device_list, ):
        super().__init__()
        self.device_list = device_list

        self.two_gpus_for_last_block = len(device_list) > 4

        block_config = parse_layer_string(block_config_str)
        channel_config = parse_channel_string(channel_config_str)
        blocks = []
        for _, (res, up_rate) in enumerate(block_config):
            if isinstance(res, tuple):
                # Denotes transition to another resolution
                res1, res2 = res
                blocks.append(
                    nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)
                )
                continue

            if up_rate is not None:
                blocks.append(nn.Upsample(scale_factor=up_rate, mode="nearest"))
                continue

            in_channel = channel_config[res]
            use_3x3 = res > 1
            blocks.append(
                ResBlock(
                    in_channel,
                    int(0.5 * in_channel),
                    in_channel,
                    down_rate=None,
                    residual=True,
                    use_3x3=use_3x3,
                )
            )
        # TODO: If the training is unstable try using scaling the weights
        num_seq = len(self.device_list)
        if self.two_gpus_for_last_block:
            num_seq -= 3
            self.last_block = blocks.pop()

        self.split_blocks = make_split_blocks(self, blocks, num_seq, n=1.2)
        self.last_conv = nn.Conv2d(channel_config[input_res], 3, 3, stride=1, padding=1)

    def forward(self, input):
        # Only executed at the very first forward pass
        self.move_layers_to_gpu()

        input = input.to(self.device_list[0])
        x = distributed_forwarding(input, self.device_list, self.split_blocks)
        if self.two_gpus_for_last_block:
            x = self.last_block(x.to(self.device_list[-3]))
        x = self.last_conv(x.to(self.device_list[-1]))
        return torch.sigmoid(x)

    def move_layers_to_gpu(self):
        for i, block in enumerate(self.split_blocks):
            block.to(self.device_list[i])

        if self.two_gpus_for_last_block:
            self.last_block.c1.to(self.device_list[-3])
            self.last_block.c2.to(self.device_list[-3])
            self.last_block.c3.to(self.device_list[-2])
            self.last_block.c4.to(self.device_list[-2])
        else:
            self.split_blocks[-1].to(self.device_list[-1])

        self.last_conv.to(self.device_list[-1])


# Implementation of the Resnet-VAE using a ResNet backbone as encoder
# and Upsampling blocks as the decoder
class VAE(pl.LightningModule):
    def __init__(
        self,
        input_res,
        enc_block_str,
        dec_block_str,
        enc_channel_str,
        dec_channel_str,
        alpha=1.0,
        lr=1e-4,
        contrastive=False,
        c_weight=1.0,
        max_c_weight=10.0,
        decay_c_rate=1e-5,
        batch_size=16,
        masking_prob=0.0,
        square=25,
        pid=False,
        dpi=False,
        c_max=None,
        gpu_list=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_res = input_res
        self.enc_block_str = enc_block_str
        self.dec_block_str = dec_block_str
        self.enc_channel_str = enc_channel_str
        self.dec_channel_str = dec_channel_str
        self.alpha = alpha
        self.lr = lr
        self.contrastive = contrastive
        self.masking_prob = masking_prob
        self.square = square
        self.pid = pid
        self.dpi = dpi

        if gpu_list is None:
            num_gpus = torch.cuda.device_count()
            gpu_list = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        else:
            num_gpus = len(gpu_list)

        self.vae_gpu = gpu_list[0]
        print(f"### num gpu {num_gpus}")
        if num_gpus == 1:
            self.enc_gpus = gpu_list
            self.dec_gpus = gpu_list
        else:
            self.enc_gpus = gpu_list[: len(gpu_list) // 2]
            self.dec_gpus = gpu_list[len(gpu_list) // 2 :]

        self.batch_size = batch_size
        self.mse_loss = nn.MSELoss(reduction="sum").to(self.vae_gpu)

        # Encoder architecture

        self.enc = Encoder(self.enc_block_str, self.enc_channel_str, self.enc_gpus)

        # Decoder Architecture

        self.dec = Decoder(
            self.input_res, self.dec_block_str, self.dec_channel_str, self.dec_gpus
        )

        # Contrastive tools:
        if self.contrastive:
            self.c_weight = lambda step: c_weight + (max_c_weight - c_weight) * np.exp(
                -decay_c_rate * step
            )

            self.nt_xent = NT_Xent(self.batch_size, 0.5).to(self.vae_gpu)

        if self.pid:
            self.pid_controller = PIDControlWrapper(c_max, device)
        if self.dpi:
            self.dpi_controller = IncrementalPIControllerWrapper(c_max, device)

        self.scripted_transforms = create_transforms(self.vae_gpu)

    def encode(self, x):
        mu, logvar = self.enc(x)
        return mu.to(self.vae_gpu), logvar.to(self.vae_gpu)

    def decode(self, z):
        return self.dec(z).to(self.vae_gpu)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if self.contrastive:
            return reparametrize_non_diag(mu, std, eps)
        return mu + eps * std

    def compute_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, z):
        # Only sample during inference
        decoder_out = self.decode(z)
        return decoder_out

    def forward_recons(self, x):
        # For generating reconstructions during inference
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoder_out = self.decode(z)
        return decoder_out

    def training_step(self, batch, batch_idx):
        batch = batch.to(self.vae_gpu)
        x = batch
        self.last_batch = batch

        if self.contrastive:
            transforms = [t(x) for t in self.scripted_transforms]

            recons_loss = 0
            kl_loss = 0
            z = []

            for x in transforms:
                # Encoder
                #mask = random_zero_square_batch(x, self.square).to(x.device)
                #x = x * mask
                mu, logvar = self.encode(x)

                # Reparameterization Trick
                _z = self.reparameterize(mu, logvar)
                z.append(torch.squeeze(_z))

                # Decoder
                decoder_out = self.decode(_z)

                # Compute loss
                recons_loss += self.mse_loss(decoder_out, x)
                kl_loss += self.compute_kl(mu, logvar)

            recons_loss /= 2 * self.batch_size
            kl_loss /= 2 * self.batch_size

            c_loss = self.nt_xent(z[0], z[1])

            total_loss = (
                recons_loss
                + self.alpha * kl_loss
                + self.c_weight(self.global_step) * c_loss
            )

            self.log("Recons Loss", recons_loss, prog_bar=True)
            self.log("Kl Loss", kl_loss, prog_bar=True)
            self.log("C Loss", c_loss, prog_bar=True)
            self.log("Total Loss", total_loss)
            self.log("c_weight", self.c_weight(self.global_step))

            return total_loss

        else:
            # Encoder
            mu, logvar = self.encode(x)

            # Reparameterization Trick
            z = self.reparameterize(mu, logvar)

            # Decoder
            decoder_out = self.decode(z)

            # Compute loss
            recons_loss = self.mse_loss(decoder_out, x)
            kl_loss = self.compute_kl(mu, logvar)

            recons_loss /= self.batch_size
            kl_loss /= self.batch_size

            self.log("Recons Loss", recons_loss, prog_bar=True)
            self.log("Kl Loss", kl_loss, prog_bar=True)

            if self.pid:
                alpha = self.pid_controller.pid(kl_loss.item(), self.global_step)
            elif self.dpi:
                alpha = self.dpi_controller.pid(kl_loss.item())
            else:
                alpha = self.alpha

            total_loss = recons_loss + alpha * kl_loss
            self.log("Total Loss", total_loss)
            return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    enc_block_config_str = "128x1,128d2,128t64,64x3,64d2,64t32,32x3,32d2,32t16,16x7,16d2,16t8,8x3,8d2,8t4,4x3,4d4,4t1,1x2"
    enc_channel_config_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024"

    dec_block_config_str = "1x1,1u4,1t4,4x2,4u2,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
    dec_channel_config_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024"

    input_res = 128

    vae = VAE(
        input_res,
        enc_block_config_str,
        dec_block_config_str,
        enc_channel_config_str,
        dec_channel_config_str,
        # contrastive=True,
        # pid=True,
    )

    sample = torch.randn(16, 3, input_res, input_res)
    out = vae.training_step(sample, 0)
    print(vae)
    print(out.shape)
