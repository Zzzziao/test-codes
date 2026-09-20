"""CUDA-only SID-UNN with joint optimization of the network and WLS parameters.

This experimental entry point preserves the L1 reconstruction loss, EMA output
smoothing, input-noise regularization, and optional total-variation penalty from
SID_WLS.py. The WLS solve is differentiable, so alpha and lambda are learned
together with the untrained image generator.
"""

from __future__ import print_function

import copy
import os
import argparse
import json
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F

from models import get_net
from utils.common_utils import get_noise, np_to_torch, plot_image_grid, plot_image_grid_final, torch_to_np, tv_loss
from utils.WLS_utils_gpu import (
    atmospheric_scattering_model,
    dark_channel,
    estimate_atmospheric_light,
    inverse_softplus,
    transmission_estimate_wls,
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# Input and model configuration
image_path = "hazy input/H1.jpg"
input_mode = "noise"
input_depth = 3
padding_mode = "reflection"
num_iterations = 2600
network_learning_rate = 1e-2
physical_parameter_learning_rate = 1e-3

# Physical-prior configuration
patch_size = 7
initial_alpha = 4.0
initial_lambda = 0.35
min_alpha = 1.0
omega = 0.95
min_transmission = 0.1
wls_epsilon = 1e-4
cg_iterations = 2000

# DIP regularization and output smoothing
reg_noise_std = 1.0 / 30.0
tv_weight = 0.0
ema_weight = 0.90

# Monitoring
plot = True
show_every = 100
check_every = 100
psnr_drop_threshold = 5.0
figure_size = 5


def psnr(reference, estimate):
    """Calculate PSNR for tensors whose expected range is [0, 1]."""
    mse = F.mse_loss(estimate, reference)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default=image_path)
    parser.add_argument('--iterations', type=int, default=num_iterations)
    parser.add_argument('--max-size', type=int, default=0, help='Optional longest image side for a smaller experiment')
    parser.add_argument('--output-dir', default='results/sid_joint')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index')
    parser.add_argument('--cg-iterations', type=int, default=cg_iterations)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError(
            'SID_gpu.py requires a CUDA-enabled PyTorch build and an available NVIDIA GPU. '
            'The current PyTorch reports torch.cuda.is_available() == False.'
        )
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        parser.error('--gpu must identify an available CUDA device (found {}).'.format(torch.cuda.device_count()))
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    image_pil = Image.open(args.image).convert('RGB')
    if args.max_size:
        image_pil.thumbnail((args.max_size, args.max_size), Image.Resampling.LANCZOS)
    image_np = np.asarray(image_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
    hazy_image = np_to_torch(image_np).to(device=device, dtype=torch.float32)
    height, width = hazy_image.shape[-2:]
    # Five downsampling stages require adequate dimensions. Pad only the noise
    # input and crop the network output, preserving every original image pixel.
    network_size = (max(64, (height + 31) // 32 * 32), max(64, (width + 31) // 32 * 32))

    # Atmospheric light remains a fixed physical estimate, matching SID_WLS.
    initial_dark_channel = dark_channel(hazy_image, patch_size)
    atmospheric_light = estimate_atmospheric_light(hazy_image, initial_dark_channel)

    network = get_net(
        input_depth,
        "skip",
        padding_mode,
        skip_n33d=128,
        skip_n33u=128,
        skip_n11=4,
        num_scales=5,
        upsample_mode="bilinear",
    ).to(device)

    network_input = get_noise(
        input_depth,
        input_mode,
        network_size,
    ).to(device)
    saved_network_input = network_input.detach().clone()
    input_noise = network_input.detach().clone()

    # Raw parameters are unconstrained. Softplus below converts them to valid
    # positive physical values while retaining useful gradients.
    alpha_raw = torch.nn.Parameter(
        torch.tensor(inverse_softplus(initial_alpha - min_alpha), device=device)
    )
    lambda_raw = torch.nn.Parameter(
        torch.tensor(inverse_softplus(initial_lambda), device=device)
    )

    optimizer = torch.optim.Adam(
        [
            {"params": network.parameters(), "lr": network_learning_rate},
            {"params": [alpha_raw, lambda_raw], "lr": physical_parameter_learning_rate},
        ]
    )
    l1_loss = torch.nn.L1Loss()

    output_ema = None
    previous_checkpoint_psnr = None
    checkpoint = None

    print("Device: {} ({})".format(device, torch.cuda.get_device_name(device)))
    print("Number of network parameters: {}".format(sum(p.numel() for p in network.parameters())))

    history = []
    for iteration in range(args.iterations):
        optimizer.zero_grad()

        if reg_noise_std > 0:
            current_input = saved_network_input + input_noise.normal_() * reg_noise_std
        else:
            current_input = saved_network_input

        clear_image = network(current_input)[..., :height, :width]

        alpha = min_alpha + F.softplus(alpha_raw)
        lambda_ = F.softplus(lambda_raw)
        transmission = transmission_estimate_wls(
            hazy_image,
            atmospheric_light,
            patch_size,
            lambda_,
            alpha,
            omega=omega,
            epsilon=wls_epsilon,
            cg_iterations=args.cg_iterations,
        )
        reconstructed_hazy = atmospheric_scattering_model(
            clear_image,
            transmission,
            atmospheric_light,
            min_transmission=min_transmission,
        )

        reconstruction_loss = l1_loss(reconstructed_hazy, hazy_image)
        regularization_loss = tv_weight * tv_loss(clear_image) if tv_weight > 0 else clear_image.new_zeros(())
        total_loss = reconstruction_loss + regularization_loss
        total_loss.backward()
        for name, parameter in [('alpha_raw', alpha_raw), ('lambda_raw', lambda_raw)]:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError('Missing or nonfinite gradient: ' + name)

        alpha_gradient = alpha_raw.grad.detach().item() if alpha_raw.grad is not None else float("nan")
        lambda_gradient = lambda_raw.grad.detach().item() if lambda_raw.grad is not None else float("nan")
        optimizer.step()

        with torch.no_grad():
            if output_ema is None:
                output_ema = clear_image.detach().clone()
            else:
                output_ema.mul_(ema_weight).add_(clear_image.detach(), alpha=1.0 - ema_weight)

            reconstruction_psnr = psnr(hazy_image, reconstructed_hazy).item()

        history.append(dict(iteration=iteration, loss=total_loss.item(), psnr=reconstruction_psnr,
                            alpha=alpha.item(), lambda_=lambda_.item(), alpha_grad=alpha_gradient, lambda_grad=lambda_gradient))
        completed_iterations = iteration + 1
        if completed_iterations % show_every == 0 or completed_iterations == args.iterations:
            print(
                "Iteration {:05d}  Loss {:.6f}  alpha {:.6f}  lambda {:.6f}".format(
                    completed_iterations,
                    total_loss.item(),
                    alpha.item(),
                    lambda_.item(),
                )
            )
            current_output = np.clip(torch_to_np(output_ema), 0, 1)
            Image.fromarray(
                (current_output.transpose(1, 2, 0) * 255).round().astype(np.uint8)
            ).save(output_dir / 'out_{:05d}.png'.format(completed_iterations))

            if plot and not args.no_plot:
                plot_image_grid(
                    [
                        np.clip(torch_to_np(clear_image), 0, 1),
                        current_output,
                        np.clip(torch_to_np(transmission), 0, 1),
                    ],
                    factor=figure_size,
                    nrow=1,
                )

        # Checkpoint only at the requested interval. The old condition ran on
        # almost every iteration because it omitted the equality-to-zero test.
        if iteration > 0 and iteration % check_every == 0:
            if (
                previous_checkpoint_psnr is not None
                and reconstruction_psnr < previous_checkpoint_psnr - psnr_drop_threshold
                and checkpoint is not None
            ):
                print("PSNR dropped sharply; restoring the previous checkpoint.")
                network.load_state_dict(checkpoint["network"])
                alpha_raw.data.copy_(checkpoint["alpha_raw"])
                lambda_raw.data.copy_(checkpoint["lambda_raw"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                output_ema = checkpoint['output_ema'].clone()
            else:
                checkpoint = {
                    "network": copy.deepcopy(network.state_dict()),
                    "alpha_raw": alpha_raw.detach().clone(),
                    "lambda_raw": lambda_raw.detach().clone(),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "output_ema": output_ema.clone(),
                }
                previous_checkpoint_psnr = reconstruction_psnr

    final_output = output_ema if output_ema is not None else network(saved_network_input).detach()
    result = np.clip(torch_to_np(final_output), 0, 1)
    Image.fromarray((result.transpose(1, 2, 0) * 255).round().astype(np.uint8)).save(output_dir / 'dehazed.png')
    image_pil.save(output_dir / 'hazy.png')
    comparison = np.concatenate([image_np, result], axis=2)
    Image.fromarray((comparison.transpose(1, 2, 0) * 255).round().astype(np.uint8)).save(output_dir / 'comparison.png')
    (output_dir / 'history.json').write_text(json.dumps(history, indent=2))
    print('Saved results to', output_dir.resolve())
    if plot and not args.no_plot:
        plot_image_grid_final([result, image_np], factor=13, nrow=1)


if __name__ == "__main__":
    main()
