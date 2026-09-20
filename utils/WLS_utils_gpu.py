"""CUDA-only differentiable dark-channel and weighted-least-squares utilities.

The WLS solve in this module stays entirely inside PyTorch. In particular, it
does not convert tensors to NumPy/CuPy or call SciPy, so autograd can propagate
the reconstruction loss back to the learnable WLS parameters ``alpha`` and
``lambda_``.
"""

import math

import torch
import torch.nn.functional as F


def _check_image(image):
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected an RGB tensor with shape [batch, 3, height, width].")
    if not image.is_cuda:
        raise ValueError("WLS GPU variant requires CUDA tensors; got {}.".format(image.device))


def dark_channel(image, patch_size=7):
    """Return the local RGB minimum used by the dark-channel prior.

    Negated max pooling is equivalent to minimum pooling. This corrects the
    previous implementation, which accidentally selected a local maximum.
    """
    _check_image(image)
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer.")

    channel_min = image.min(dim=1, keepdim=True).values
    return -F.max_pool2d(
        -channel_min,
        kernel_size=patch_size,
        stride=1,
        padding=patch_size // 2,
    )


@torch.no_grad()
def estimate_atmospheric_light(image, dark=None, top_fraction=1e-3):
    """Estimate one RGB atmospheric-light vector per image.

    The estimate is the mean RGB value of pixels belonging to the brightest
    fraction of the dark channel. Atmospheric light is intentionally fixed in
    SID-UNN; only the WLS parameters and network parameters are optimized.
    """
    _check_image(image)
    if dark is None:
        dark = dark_channel(image)

    batch, channels, height, width = image.shape
    pixel_count = height * width
    candidate_count = max(1, int(math.ceil(pixel_count * top_fraction)))

    dark_flat = dark.reshape(batch, pixel_count)
    candidate_indices = dark_flat.topk(candidate_count, dim=1).indices
    image_flat = image.reshape(batch, channels, pixel_count)
    gather_indices = candidate_indices.unsqueeze(1).expand(-1, channels, -1)
    candidates = image_flat.gather(2, gather_indices)
    return candidates.mean(dim=2).view(batch, channels, 1, 1)


def initial_transmission(image, atmospheric_light, patch_size=7, omega=0.95):
    """Compute the unsmoothed DCP transmission estimate."""
    _check_image(image)
    safe_airlight = atmospheric_light.clamp_min(1e-6)
    normalized_image = image / safe_airlight
    return 1.0 - omega * dark_channel(normalized_image, patch_size)


def _wls_weights(guidance, lambda_, alpha, epsilon):
    """Calculate positive horizontal and vertical WLS edge penalties."""
    log_guidance = torch.log(guidance + 1e-10)
    dx = log_guidance[..., :, :-1] - log_guidance[..., :, 1:]
    dy = log_guidance[..., :-1, :] - log_guidance[..., 1:, :]

    # Clamp the gradient magnitude before exponentiation. At an exact zero,
    # d(|g|**alpha)/d(alpha) contains log(0) and can produce a NaN gradient.
    safe_dx = dx.abs().clamp_min(1e-8)
    safe_dy = dy.abs().clamp_min(1e-8)
    weight_x = lambda_ / (safe_dx.pow(alpha) + epsilon)
    weight_y = lambda_ / (safe_dy.pow(alpha) + epsilon)
    return weight_x, weight_y


def _apply_wls_operator(value, weight_x, weight_y):
    """Apply A(value) for the WLS system without explicitly building A."""
    horizontal_flux = weight_x * (value[..., :, :-1] - value[..., :, 1:])
    vertical_flux = weight_y * (value[..., :-1, :] - value[..., 1:, :])

    horizontal = F.pad(horizontal_flux, (0, 1)) - F.pad(horizontal_flux, (1, 0))
    vertical = F.pad(vertical_flux, (0, 0, 0, 1)) - F.pad(vertical_flux, (0, 0, 1, 0))
    return value + horizontal + vertical


def _wls_diagonal(weight_x, weight_y):
    """Return the diagonal of the WLS matrix for Jacobi preconditioning."""
    horizontal = F.pad(weight_x, (0, 1)) + F.pad(weight_x, (1, 0))
    vertical = F.pad(weight_y, (0, 0, 0, 1)) + F.pad(weight_y, (0, 0, 1, 0))
    return 1.0 + horizontal + vertical


def _batch_inner(left, right):
    return (left * right).sum(dim=(-2, -1), keepdim=True)


def differentiable_pcg(
    right_hand_side,
    apply_operator,
    diagonal,
    iterations=2000,
    numerical_epsilon=1e-12,
    tolerance=1e-8,
):
    """Solve each batch member on CUDA without per-iteration host synchronization."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    tiny = torch.finfo(right_hand_side.dtype).tiny
    target = tolerance ** 2 * _batch_inner(right_hand_side, right_hand_side).clamp_min(tiny)
    solution = right_hand_side.clone()
    residual = right_hand_side - apply_operator(solution)
    preconditioned = residual / diagonal.clamp_min(numerical_epsilon)
    direction = preconditioned
    residual_product = _batch_inner(residual, preconditioned)

    for _ in range(iterations):
        active = _batch_inner(residual, residual) > target
        operator_direction = apply_operator(direction)
        denominator = _batch_inner(direction, operator_direction).clamp_min(tiny)
        step = torch.where(active, residual_product / denominator, 0.0)

        solution = solution + step * direction
        residual = residual - step * operator_direction
        next_preconditioned = residual / diagonal.clamp_min(numerical_epsilon)
        next_residual_product = _batch_inner(residual, next_preconditioned)
        conjugate_scale = torch.where(active, next_residual_product / residual_product.clamp_min(tiny), 0.0)

        direction = next_preconditioned + conjugate_scale * direction
        preconditioned = next_preconditioned
        residual_product = next_residual_product

    relative_residual = (_batch_inner(right_hand_side - apply_operator(solution), right_hand_side - apply_operator(solution)) / _batch_inner(right_hand_side, right_hand_side).clamp_min(tiny)).sqrt()
    if (relative_residual > tolerance * 2).any():
        raise RuntimeError("WLS PCG did not converge: relative residual {:.3e}; increase cg_iterations".format(relative_residual.max().item()))
    return solution


class _WLSSolve(torch.autograd.Function):
    """Implicit differentiation avoids storing every CG iteration for backward."""

    @staticmethod
    def forward(ctx, rhs, weight_x, weight_y, iterations):
        result = differentiable_pcg(rhs, lambda x: _apply_wls_operator(x, weight_x, weight_y),
                                    _wls_diagonal(weight_x, weight_y), iterations)
        ctx.save_for_backward(result, weight_x, weight_y)
        ctx.iterations = iterations
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, weight_x, weight_y = ctx.saved_tensors
        adjoint = differentiable_pcg(grad_output, lambda x: _apply_wls_operator(x, weight_x, weight_y),
                                     _wls_diagonal(weight_x, weight_y), ctx.iterations)
        grad_x = -(adjoint[..., :, :-1] - adjoint[..., :, 1:]) * (result[..., :, :-1] - result[..., :, 1:])
        grad_y = -(adjoint[..., :-1, :] - adjoint[..., 1:, :]) * (result[..., :-1, :] - result[..., 1:, :])
        return adjoint, grad_x, grad_y, None


def differentiable_wls_filter(
    transmission,
    lambda_,
    alpha,
    epsilon=1e-4,
    cg_iterations=2000,
):
    """Smooth a transmission map while retaining gradients to alpha and lambda."""
    if transmission.ndim != 4 or transmission.shape[1] != 1:
        raise ValueError("Expected transmission with shape [batch, 1, height, width].")
    if not transmission.is_cuda:
        raise ValueError("WLS GPU variant requires a CUDA transmission tensor; got {}.".format(transmission.device))
    if not lambda_.is_cuda or not alpha.is_cuda:
        raise ValueError("WLS GPU variant requires CUDA alpha and lambda tensors.")
    if not (transmission.device == lambda_.device == alpha.device):
        raise ValueError("Transmission, alpha, and lambda must be on the same CUDA device.")

    if epsilon <= 0 or not torch.isfinite(transmission).all() or (transmission < 0).any():
        raise ValueError("WLS requires finite nonnegative transmission and positive epsilon.")
    # High edge weights at alpha=4 make float32 residuals unreliable.
    rhs = transmission.double()
    weight_x, weight_y = _wls_weights(rhs, lambda_, alpha, epsilon)
    return _WLSSolve.apply(rhs, weight_x, weight_y, cg_iterations).to(transmission.dtype)


def transmission_estimate_wls(
    image,
    atmospheric_light,
    patch_size,
    lambda_,
    alpha,
    omega=0.95,
    epsilon=1e-4,
    cg_iterations=2000,
):
    """Create a DCP transmission map and refine it with differentiable WLS."""
    transmission = initial_transmission(image, atmospheric_light, patch_size, omega)
    return differentiable_wls_filter(
        transmission,
        lambda_,
        alpha,
        epsilon=epsilon,
        cg_iterations=cg_iterations,
    )


def atmospheric_scattering_model(clear_image, transmission, atmospheric_light, min_transmission=0.1):
    """Reconstruct the hazy image I = J*t + A*(1-t)."""
    _check_image(clear_image)
    bounded_transmission = transmission.clamp(min=min_transmission, max=1.0)
    return clear_image * bounded_transmission + atmospheric_light * (1.0 - bounded_transmission)


def inverse_softplus(value):
    """Return an unconstrained scalar whose softplus equals ``value``."""
    if value <= 0:
        raise ValueError("The requested positive initial value must be greater than zero.")
    return math.log(math.expm1(value))
