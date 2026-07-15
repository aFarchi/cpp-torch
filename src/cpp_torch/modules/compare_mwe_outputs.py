"""Compare decoded and encoded outputs from mwe13.py and mwe14.py.

Usage:
    pixi run python compare_mwe_outputs.py

Optional args:
    --rtol FLOAT Relative tolerance for torch.allclose (default: 1e-6)
    --atol FLOAT Absolute tolerance for torch.allclose (default: 1e-7)
"""
from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare decoded/encoded tensors.")
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


def compare_pair(name: str, path_a: str, path_b: str,
                 rtol: float, atol: float) -> None:
    a = torch.load(path_a, map_location="cpu")
    b = torch.load(path_b, map_location="cpu")

    if a.shape != b.shape:
        raise AssertionError(
            f"[{name}] Shape mismatch: {path_a} has {tuple(a.shape)}, "
            f"{path_b} has {tuple(b.shape)}"
        )

    diff = (a - b).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_l2 = (diff.norm() / a.norm()).item() if a.norm().item() != 0.0 else float("inf")
    is_close = torch.allclose(a, b, rtol=rtol, atol=atol)

    print(f"[{name}] A: {path_a}")
    print(f"[{name}] B: {path_b}")
    print(f"[{name}] shape: {tuple(a.shape)}")
    print(f"[{name}] max_abs_diff: {max_abs:.6e}")
    print(f"[{name}] mean_abs_diff: {mean_abs:.6e}")
    print(f"[{name}] rel_l2_diff: {rel_l2:.6e}")
    print(f"[{name}] allclose(rtol={rtol}, atol={atol}): {is_close}")

    assert is_close, (
        f"[{name}] Outputs are not allclose. "
        f"max_abs_diff={max_abs:.6e}, mean_abs_diff={mean_abs:.6e}, "
        f"rel_l2_diff={rel_l2:.6e}, rtol={rtol}, atol={atol}"
    )


def main() -> None:
    args = parse_args()
    compare_pair(
        name="decoded",
        path_a="output/decoded_output_13.pt",
        path_b="output/decoded_output_14.pt",
        rtol=args.rtol,
        atol=args.atol,
    )
    compare_pair(
        name="encoded",
        path_a="output/encoded_from_decoded_13.pt",
        path_b="output/encoded_from_decoded_14.pt",
        rtol=args.rtol,
        atol=args.atol,
    )
    print("PASS: decoded and encoded outputs are allclose.")


if __name__ == "__main__":
    main()
