"""Controlled adversarial-like perturbation simulations for defensive research."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np


ATTACK_TYPES = ("gaussian", "sinusoidal", "checkerboard", "dct", "fft")


def _validate(image: np.ndarray, epsilon: float) -> np.ndarray:
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a non-empty three-channel image")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number")
    return image.astype(np.float32)


def _finish(image: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return np.clip(image + delta, 0, 255).round().astype(np.uint8)


def gaussian_perturbation(
    image: np.ndarray, epsilon: float, rng: np.random.Generator
) -> np.ndarray:
    source = _validate(image, epsilon)
    delta = rng.normal(0.0, epsilon, size=source.shape).astype(np.float32)
    delta = np.clip(delta, -2.0 * epsilon, 2.0 * epsilon)
    return _finish(source, delta)


def sinusoidal_perturbation(
    image: np.ndarray, epsilon: float, rng: np.random.Generator
) -> np.ndarray:
    source = _validate(image, epsilon)
    height, width = source.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    frequency = int(rng.integers(12, 25))
    phase = float(rng.uniform(0, 2 * np.pi))
    angle = float(rng.uniform(0, np.pi))
    coordinate = xx * np.cos(angle) + yy * np.sin(angle)
    pattern = epsilon * np.sin(2 * np.pi * frequency * coordinate / max(height, width) + phase)
    delta = np.repeat(pattern[..., None], 3, axis=2).astype(np.float32)
    return _finish(source, delta)


def checkerboard_perturbation(
    image: np.ndarray, epsilon: float, rng: np.random.Generator
) -> np.ndarray:
    source = _validate(image, epsilon)
    height, width = source.shape[:2]
    block = int(rng.integers(1, 5))
    yy, xx = np.mgrid[:height, :width]
    pattern = (((xx // block + yy // block) % 2) * 2 - 1).astype(np.float32)
    delta = np.repeat((epsilon * pattern)[..., None], 3, axis=2)
    return _finish(source, delta)


def _normalized_frequency_pattern(
    height: int,
    width: int,
    rng: np.random.Generator,
    transform: str,
) -> np.ndarray:
    """Create a unit-amplitude spatial pattern from sparse frequency coefficients."""
    if transform == "dct":
        y = int(rng.integers(max(1, height // 3), height))
        x = int(rng.integers(max(1, width // 3), width))
        rows = np.arange(height, dtype=np.float32)[:, None]
        columns = np.arange(width, dtype=np.float32)[None, :]
        # A single 2D DCT-II basis works for both even and odd dimensions.
        vertical = np.cos(np.pi * (2 * rows + 1) * y / (2 * height))
        horizontal = np.cos(np.pi * (2 * columns + 1) * x / (2 * width))
        pattern = (vertical * horizontal).astype(np.float32)
    elif transform == "fft":
        spectrum = np.zeros((height, width), dtype=np.complex64)
        y = int(rng.integers(max(1, height // 4), max(2, 3 * height // 4)))
        x = int(rng.integers(max(1, width // 4), max(2, 3 * width // 4)))
        phase = float(rng.uniform(0, 2 * np.pi))
        value = np.exp(1j * phase)
        spectrum[y, x] = value
        spectrum[-y % height, -x % width] = np.conjugate(value)
        pattern = np.fft.ifft2(spectrum).real.astype(np.float32)
    else:
        raise ValueError(f"unsupported transform: {transform}")

    maximum = float(np.max(np.abs(pattern)))
    if maximum <= np.finfo(np.float32).eps:
        raise ValueError(f"failed to create {transform} perturbation pattern")
    return pattern / maximum


def dct_perturbation(
    image: np.ndarray, epsilon: float, rng: np.random.Generator
) -> np.ndarray:
    source = _validate(image, epsilon)
    pattern = _normalized_frequency_pattern(*source.shape[:2], rng, "dct")
    delta = np.repeat((epsilon * pattern)[..., None], 3, axis=2)
    return _finish(source, delta)


def fft_perturbation(
    image: np.ndarray, epsilon: float, rng: np.random.Generator
) -> np.ndarray:
    source = _validate(image, epsilon)
    pattern = _normalized_frequency_pattern(*source.shape[:2], rng, "fft")
    delta = np.repeat((epsilon * pattern)[..., None], 3, axis=2)
    return _finish(source, delta)


PERTURBATION_FUNCTIONS: dict[
    str, Callable[[np.ndarray, float, np.random.Generator], np.ndarray]
] = {
    "gaussian": gaussian_perturbation,
    "sinusoidal": sinusoidal_perturbation,
    "checkerboard": checkerboard_perturbation,
    "dct": dct_perturbation,
    "fft": fft_perturbation,
}


def generate_perturbation(
    image: np.ndarray, attack_type: str, epsilon: float, seed: int
) -> np.ndarray:
    """Generate one deterministic controlled perturbation simulation."""
    try:
        function = PERTURBATION_FUNCTIONS[attack_type]
    except KeyError as exc:
        raise ValueError(f"unknown attack type: {attack_type}") from exc
    return function(image, epsilon, np.random.default_rng(seed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one controlled adversarial-like image simulation."
    )
    parser.add_argument("input", type=Path, help="source image path")
    parser.add_argument("output", type=Path, help="output PNG path")
    parser.add_argument("--attack-type", choices=ATTACK_TYPES, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Could not decode input image: {args.input}", file=sys.stderr)
        return 1
    try:
        manipulated = generate_perturbation(
            image, args.attack_type, args.epsilon, args.seed
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), manipulated):
        print(f"Could not write output image: {args.output}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
