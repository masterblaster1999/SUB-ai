#!/usr/bin/env python3
"""generate_multidigit_dataset.py

Synthetic multi-digit dataset generator for SUB ai.

Why this exists
---------------
The original SUB ai number model is a *single-digit* MNIST classifier.
For multi-digit numbers, this repository now supports contour segmentation
and per-digit classification. That works well, but it is also useful to have
multi-digit training data (e.g., for future sequence models or evaluating the
segmentation pipeline).

This script composes multi-digit images by sampling MNIST digits and placing
them side-by-side with random spacing, scaling, and small vertical jitter.

Outputs
-------
* PNG images in the output directory
* labels.csv with (filename,label)

Example
-------
python generate_multidigit_dataset.py \
  --output data/multidigit \
  --samples 5000 \
  --min-digits 2 \
  --max-digits 5
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class GenerationConfig:
    output_dir: str
    samples: int
    min_digits: int
    max_digits: int
    seed: int
    spacing_min: int
    spacing_max: int
    scale_min: float
    scale_max: float
    jitter: int
    noise_sigma: float
    noise_prob: float
    blur_prob: float


def _load_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST (train split) as uint8 images."""

    from tensorflow import keras

    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    # MNIST is already 28x28, uint8 in [0,255]
    return x_train.astype(np.uint8), y_train.astype(np.int64)


def _pick_digit_image(x: np.ndarray, y: np.ndarray, digit: int, rng: np.random.Generator) -> np.ndarray:
    idxs = np.where(y == digit)[0]
    if idxs.size == 0:
        raise ValueError(f"No MNIST samples for digit {digit}")
    idx = int(rng.choice(idxs))
    return x[idx]


def _compose_number_image(
    digits: List[int],
    x_mnist: np.ndarray,
    y_mnist: np.ndarray,
    cfg: GenerationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compose a multi-digit grayscale image (digits white on black)."""

    spacing = int(rng.integers(cfg.spacing_min, cfg.spacing_max + 1))

    # Add a small padding around the composed number.
    pad = 6
    base_h = 28

    # Conservative width estimate (scaling may increase a bit)
    max_scale = max(cfg.scale_max, 1.0)
    est_digit_w = int(round(28 * max_scale))
    canvas_w = pad * 2 + len(digits) * est_digit_w + (len(digits) - 1) * spacing
    canvas_h = pad * 2 + int(round(base_h * max_scale))
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    x_cursor = pad

    for d in digits:
        img = _pick_digit_image(x_mnist, y_mnist, d, rng)

        # Random scaling
        scale = float(rng.uniform(cfg.scale_min, cfg.scale_max))
        new_w = max(1, int(round(img.shape[1] * scale)))
        new_h = max(1, int(round(img.shape[0] * scale)))
        digit_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Small vertical jitter
        y_center = (canvas_h - new_h) // 2
        y_off = int(np.clip(y_center + int(rng.integers(-cfg.jitter, cfg.jitter + 1)), 0, canvas_h - new_h))

        # Paste using max to preserve strokes when overlapping slightly
        x_end = min(canvas_w, x_cursor + new_w)
        digit_crop = digit_img[:, : x_end - x_cursor]
        canvas[y_off : y_off + new_h, x_cursor:x_end] = np.maximum(
            canvas[y_off : y_off + new_h, x_cursor:x_end],
            digit_crop,
        )

        x_cursor += new_w + spacing

    # Optional noise
    if cfg.noise_sigma > 0 and rng.random() < cfg.noise_prob:
        noise = rng.normal(0.0, cfg.noise_sigma, size=canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Optional blur
    if rng.random() < cfg.blur_prob:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    return canvas


def generate(cfg: GenerationConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    print("Loading MNIST...")
    x_mnist, y_mnist = _load_mnist()

    labels_path = os.path.join(cfg.output_dir, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])

        for i in range(cfg.samples):
            n_digits = int(rng.integers(cfg.min_digits, cfg.max_digits + 1))
            digits = [int(rng.integers(0, 10)) for _ in range(n_digits)]
            label = "".join(str(d) for d in digits)

            img = _compose_number_image(digits, x_mnist, y_mnist, cfg, rng)

            filename = f"{i:06d}_{label}.png"
            path = os.path.join(cfg.output_dir, filename)
            cv2.imwrite(path, img)
            writer.writerow([filename, label])

            if (i + 1) % 500 == 0:
                print(f"Generated {i + 1}/{cfg.samples}...")

    print("\nDone!")
    print(f"Images: {cfg.output_dir}")
    print(f"Labels: {labels_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a synthetic multi-digit dataset from MNIST")
    p.add_argument("--output", dest="output_dir", default="data/multidigit", help="Output directory")
    p.add_argument("--samples", type=int, default=5000, help="Number of images to generate")
    p.add_argument("--min-digits", type=int, default=2, help="Minimum digits per image")
    p.add_argument("--max-digits", type=int, default=5, help="Maximum digits per image")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    p.add_argument("--spacing-min", type=int, default=2, help="Minimum spacing between digits")
    p.add_argument("--spacing-max", type=int, default=8, help="Maximum spacing between digits")
    p.add_argument("--scale-min", type=float, default=0.85, help="Minimum digit scale")
    p.add_argument("--scale-max", type=float, default=1.10, help="Maximum digit scale")
    p.add_argument("--jitter", type=int, default=2, help="Vertical jitter in pixels")

    p.add_argument("--noise-sigma", type=float, default=8.0, help="Gaussian noise sigma (0 disables)")
    p.add_argument("--noise-prob", type=float, default=0.35, help="Probability of adding noise")
    p.add_argument("--blur-prob", type=float, default=0.15, help="Probability of applying blur")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.min_digits < 1:
        raise SystemExit("--min-digits must be >= 1")
    if args.max_digits < args.min_digits:
        raise SystemExit("--max-digits must be >= --min-digits")
    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")

    cfg = GenerationConfig(
        output_dir=str(args.output_dir),
        samples=int(args.samples),
        min_digits=int(args.min_digits),
        max_digits=int(args.max_digits),
        seed=int(args.seed),
        spacing_min=int(args.spacing_min),
        spacing_max=int(args.spacing_max),
        scale_min=float(args.scale_min),
        scale_max=float(args.scale_max),
        jitter=int(args.jitter),
        noise_sigma=float(args.noise_sigma),
        noise_prob=float(args.noise_prob),
        blur_prob=float(args.blur_prob),
    )

    generate(cfg)


if __name__ == "__main__":
    main()
