"""Compare decoded outputs from mwe13.py and mwe14.py.

Usage:
    pixi run python compare_mwe_outputs.py

Optional args:
    --a PATH     First tensor path (default: output/decoded_output_13.pt)
    --b PATH     Second tensor path (default: output/decoded_output_14.pt)
    --rtol FLOAT Relative tolerance for torch.allclose (default: 1e-6)
    --atol FLOAT Absolute tolerance for torch.allclose (default: 1e-7)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two decoded tensors.")
    parser.add_argument(
        "--a",
        type=Path,
        default=Path("output/decoded_output_13.pt"),
        help="Path to first tensor file.",
    )
    parser.add_argument(
        "--b",
        type=Path,
        default=Path("output/decoded_output_14.pt"),
        help="Path to second tensor file.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for torch.allclose.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-7,
        help="Absolute tolerance for torch.allclose.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    a = torch.load(args.a, map_location="cpu")
    b = torch.load(args.b, map_location="cpu")

    if a.shape != b.shape:
        raise AssertionError(
            f"Shape mismatch: {args.a} has {tuple(a.shape)}, "
            f"{args.b} has {tuple(b.shape)}"
        )

    diff = (a - b).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_l2 = (diff.norm() / a.norm()).item() if a.norm().item() != 0.0 else float("inf")

    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print(f"shape: {tuple(a.shape)}")
    print(f"max_abs_diff: {max_abs:.6e}")
    print(f"mean_abs_diff: {mean_abs:.6e}")
    print(f"rel_l2_diff: {rel_l2:.6e}")
    print(f"allclose(rtol={args.rtol}, atol={args.atol}): {torch.allclose(a, b, rtol=args.rtol, atol=args.atol)}")

    assert torch.allclose(a, b, rtol=args.rtol, atol=args.atol), (
        "Decoded outputs are not allclose. "
        f"max_abs_diff={max_abs:.6e}, mean_abs_diff={mean_abs:.6e}, rel_l2_diff={rel_l2:.6e}, "
        f"rtol={args.rtol}, atol={args.atol}"
    )

    print("PASS: decoded outputs are allclose.")


if __name__ == "__main__":
    main()
