"""Standalone classic Dark Channel Prior (DCP) dehazing.

Example:
    python classic_dcp.py --image "hazy input/1449.jpg" --output results/dcp.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def dark_channel(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Compute min_RGB followed by a local minimum filter."""
    channel_minimum = image.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(channel_minimum, kernel)


def estimate_atmospheric_light(image: np.ndarray, dark: np.ndarray, fraction: float = 1e-3) -> np.ndarray:
    """Use the brightest dark-channel pixels and select the brightest RGB pixel."""
    height, width = dark.shape
    count = max(1, int(np.ceil(height * width * fraction)))
    flat_indices = np.argpartition(dark.ravel(), -count)[-count:]
    candidates = image.reshape(-1, 3)[flat_indices]
    return candidates[np.argmax(candidates.sum(axis=1))]


def estimate_transmission(image: np.ndarray, atmospheric_light: np.ndarray, patch_size: int, omega: float) -> np.ndarray:
    normalized = image / np.maximum(atmospheric_light, 1e-6)[None, None, :]
    return 1.0 - omega * dark_channel(normalized, patch_size)


def guided_filter(guidance: np.ndarray, source: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    """Grayscale guided filter used to refine the initial transmission."""
    window = (2 * radius + 1, 2 * radius + 1)
    mean_guidance = cv2.boxFilter(guidance, cv2.CV_64F, window)
    mean_source = cv2.boxFilter(source, cv2.CV_64F, window)
    correlation_guidance = cv2.boxFilter(guidance * guidance, cv2.CV_64F, window)
    correlation_cross = cv2.boxFilter(guidance * source, cv2.CV_64F, window)
    variance_guidance = correlation_guidance - mean_guidance * mean_guidance
    covariance_cross = correlation_cross - mean_guidance * mean_source
    a = covariance_cross / (variance_guidance + epsilon)
    b = mean_source - a * mean_guidance
    return cv2.boxFilter(a, cv2.CV_64F, window) * guidance + cv2.boxFilter(b, cv2.CV_64F, window)


def dehaze(image_bgr: np.ndarray, patch_size: int = 15, omega: float = 0.95,
           guided_radius: int = 60, guided_epsilon: float = 1e-3,
           min_transmission: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the dehazed BGR image, refined transmission, and atmospheric light."""
    image = image_bgr.astype(np.float64) / 255.0
    dark = dark_channel(image, patch_size)
    atmospheric_light = estimate_atmospheric_light(image, dark)
    transmission = estimate_transmission(image, atmospheric_light, patch_size, omega)
    guidance = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    transmission = guided_filter(guidance, transmission, guided_radius, guided_epsilon)
    transmission = np.clip(transmission, min_transmission, 1.0)
    clear = (image - atmospheric_light[None, None, :]) / transmission[:, :, None] + atmospheric_light[None, None, :]
    return (np.clip(clear, 0.0, 1.0) * 255.0).round().astype(np.uint8), transmission, atmospheric_light


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='Input hazy image')
    parser.add_argument('--output', required=True, help='Output dehazed image')
    parser.add_argument('--patch-size', type=int, default=15)
    parser.add_argument('--omega', type=float, default=0.95)
    parser.add_argument('--guided-radius', type=int, default=60)
    parser.add_argument('--guided-epsilon', type=float, default=1e-3)
    parser.add_argument('--min-transmission', type=float, default=0.1)
    args = parser.parse_args()
    if args.patch_size < 1 or args.patch_size % 2 == 0:
        parser.error('--patch-size must be a positive odd integer')
    if args.guided_radius < 1 or args.guided_epsilon <= 0 or not 0 < args.min_transmission <= 1:
        parser.error('guided radius/epsilon and minimum transmission must be positive')

    hazy = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if hazy is None:
        parser.error('Cannot read image: {}'.format(args.image))
    clear, _, _ = dehaze(hazy, args.patch_size, args.omega, args.guided_radius,
                         args.guided_epsilon, args.min_transmission)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), clear):
        raise RuntimeError('Cannot write image: {}'.format(output))
    print('Saved classic DCP result to {}'.format(output))


if __name__ == '__main__':
    main()
