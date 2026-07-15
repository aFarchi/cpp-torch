"""Minimal working example: decode a random latent tensor with the trained
v94 wrapper decoder (models14.py).

Self-contained: every file this script needs lives under test_mwe/
(graph/, weights/, data/, models13.py). Run with:

    /perm/pazz/conda/envs/torch/bin/python mwe.py
"""
from pathlib import Path

import numpy as np
import torch

from models14 import ZigasDecoder

#==================================================
# AF
import numpy
torch.serialization.add_safe_globals([numpy.dtype])
torch.serialization.add_safe_globals([numpy.ndarray])
torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])
#==================================================

BASE = Path(__file__).resolve().parent
GRAPH = BASE / "graph"
SHARED_Z_PATH = BASE / "output/shared_latent_z.pt"

EXP_ID = "v94"
N_SUBGRAPHS = 6  # graph hierarchy has N_SUBGRAPHS+1 = 7 levels on disk

# --- v94 architecture (configs.py EXP_CONFIGS["v94"]) ---
IN_DIM_ACTIVE = 120
OUT_DIM = 114
ENCODER_HIDDEN_DIMS = [256, 256, 192, 128, 96]
DECODER_HIDDEN_DIMS = ENCODER_HIDDEN_DIMS[::-1]
LATENT_DIM = 64
HEADS = 4
GAT_DROPOUT = 0.1
FEATURE_DROPOUT = 0.1
LATENT_DROPOUT = 0.0
LATENT_NOISE_STD = 0.0
POOLING_SCHEDULE = [2]
UNPOOLING_SCHEDULE = [3]
DECODER_MLP_RATIO = 2.0
SEED = 12345


def load_or_create_shared_latent(path: Path, n_coarse: int,
                                 latent_dim: int, seed: int) -> torch.Tensor:
    expected_shape = (n_coarse, latent_dim)
    if path.exists():
        z_cpu = torch.load(path, map_location="cpu")
        if tuple(z_cpu.shape) != expected_shape:
            raise ValueError(
                f"Shared latent at {path} has shape {tuple(z_cpu.shape)}, "
                f"expected {expected_shape}"
            )
        print(f"Loaded shared latent from {path}")
        return z_cpu

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    z_cpu = torch.randn(n_coarse, latent_dim, generator=gen, dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(z_cpu, path)
    print(f"Created shared latent at {path}")
    return z_cpu




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"Random seed: {SEED}")

# -----------------------------
# Build wrapper decoder and load trained weights
# -----------------------------
decoder = ZigasDecoder(
    base_dir=BASE,
    exp_id=EXP_ID,
    n_subgraphs=N_SUBGRAPHS,
    latent_dim=LATENT_DIM,
    out_dim=OUT_DIM,
    decoder_hidden_dims=DECODER_HIDDEN_DIMS,
    heads=HEADS,
    gat_dropout=GAT_DROPOUT,
    feature_dropout=FEATURE_DROPOUT,
    decoder_mlp_ratio=DECODER_MLP_RATIO,
    pooling_schedule=POOLING_SCHEDULE,
    unpooling_schedule=UNPOOLING_SCHEDULE,
    device=device,
)

num_params = sum(p.numel() for p in decoder.parameters())
print(f"Decoder loaded: {num_params:,} parameters")

# -----------------------------
# Random latent -> decode
# -----------------------------
n_coarse = int(decoder.gat_stage_1.edge_index.max().item()) + 1
z = load_or_create_shared_latent(SHARED_Z_PATH, n_coarse, LATENT_DIM, SEED).to(device)
print(f"Random latent shape: {tuple(z.shape)}")

with torch.no_grad():
    decoded = decoder(z)

print(f"Decoded output shape: {tuple(decoded.shape)}")
print(f"decoded stats -> mean: {decoded.mean().item():.4f}, "
      f"std: {decoded.std().item():.4f}, "
      f"min: {decoded.min().item():.4f}, max: {decoded.max().item():.4f}")

out_path = BASE / "output/decoded_output_14v2.pt"
torch.save(decoded.cpu(), out_path)
print(f"Saved decoded output to {out_path}")

print('registering buffers')
for (name, tensor) in {
    'in_forward': z,
    'out_forward': decoded,
    }.items():
    decoder.register_buffer(name, tensor)

filename = 'output/scripted_decoder.pt'
print(f'saving scripted model into "{filename}"')
scripted_model = torch.jit.script(decoder)
scripted_model.save(filename)
