"""Minimal working example: decode a random latent tensor with the trained
v94 ProgressiveGraphAutoencoder decoder (models14.py).

Self-contained: every file this script needs lives under test_mwe/
(graph/, weights/, data/, models13.py). Run with:

    /perm/pazz/conda/envs/torch/bin/python mwe.py
"""
from pathlib import Path
import re

import numpy as np
import torch

from models14 import ProgressiveDecoder

#==================================================
# AF
import numpy
torch.serialization.add_safe_globals([numpy.dtype])
torch.serialization.add_safe_globals([numpy.ndarray])
torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])
#==================================================

BASE = Path(__file__).resolve().parent
GRAPH = BASE / "graph"
WEIGHTS = BASE / "weights"
DATA = BASE / "data"
SHARED_Z_PATH = BASE / "output/shared_latent_z.pt"

EXP_ID = "v94"
N_SUBGRAPHS = 6  # graph hierarchy has N_SUBGRAPHS+1 = 7 levels on disk

# Static fields used by the decoder's per-level FiLM conditioning.
STATIC_VARS = ["sin_lat", "cos_lat", "sin_lon", "cos_lon", "lnsurfgeo", "lsm"]

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
USE_VAE = True
FREE_BITS = 0.5
ENCODER_MLP_RATIO = 4.0
DECODER_MLP_RATIO = 2.0
SEED = 12345


def clean_state_dict(state_dict, prefixes=("_orig_mod.",)):
    """Strip torch.compile's '_orig_mod.' prefix from checkpoint keys."""
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
        cleaned[new_k] = v
    return cleaned


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


def decoder_state_dict_from_full_checkpoint(full_state_dict):
    """Extract/remap decoder keys from a full autoencoder checkpoint."""
    out = {}
    for key, value in full_state_dict.items():
        if not key.startswith("decoder."):
            continue

        k = key

        # Legacy models13 key layout -> models14 ProgressiveDecoder layout.
        if k.startswith("decoder.gat_blocks."):
            m = re.match(r"^decoder\.gat_blocks\.(\d+)\.(.+)$", k)
            if m is not None:
                idx = int(m.group(1)) + 1
                k = f"decoder.gat_stage_{idx}.gat_block.{m.group(2)}"

        elif k.startswith("decoder.unpoolers."):
            m = re.match(r"^decoder\.unpoolers\.(\d+)\.(.+)$", k)
            if m is not None:
                layer_idx = m.group(1)
                k = f"decoder.unpool_and_film_{layer_idx}.unpooler.{m.group(2)}"

        elif k.startswith("decoder.static_films."):
            m = re.match(r"^decoder\.static_films\.(\d+)\.(.+)$", k)
            if m is not None:
                layer_idx = m.group(1)
                k = f"decoder.unpool_and_film_{layer_idx}.film.{m.group(2)}"

        elif k.startswith("decoder.output_proj."):
            k = k.replace("decoder.output_proj.", "decoder.output_head.output_proj.", 1)

        elif k.startswith("decoder.output_residual."):
            k = k.replace("decoder.output_residual.", "decoder.output_head.output_residual.", 1)

        # Structural static feature buffers are not persistent in models14.
        if re.match(r"^decoder\.static_feats_\d+$", k):
            continue

        out[k[len("decoder."):]] = value

    return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"Random seed: {SEED}")

# -----------------------------
# Load precomputed graph hierarchy (mirrors normal_ae.py's loading branch)
# -----------------------------
graphs = [torch.load(GRAPH / f"{EXP_ID}_edge_index_ae_{lev}.pt")
            for lev in range(N_SUBGRAPHS + 1)]
edge_attrs = torch.load(GRAPH / f"{EXP_ID}_edge_attrs.pt")
level_connections_local = torch.load(GRAPH / f"{EXP_ID}_level_connections_local.pt")
unpool_level_connections_local = torch.load(
    GRAPH / f"{EXP_ID}_unpool_level_connections_local.pt")
node_mappings = torch.load(GRAPH / f"{EXP_ID}_node_mappings.pt", weights_only=False)
pool_edge_attrs = torch.load(GRAPH / f"{EXP_ID}_pool_interlevel_edge_attrs.pt")
unpool_edge_attrs = torch.load(GRAPH / f"{EXP_ID}_unpool_interlevel_edge_attrs.pt")

graphs = [g.to(device) for g in graphs]

# -----------------------------
# Decoder static FiLM features (normal_ae.py:396-402)
# -----------------------------
statics_full = torch.tensor(
    np.stack([np.load(DATA / f"{v}.npy") for v in STATIC_VARS], axis=-1),
    dtype=torch.float32)
decoder_static_feats = [
    statics_full[torch.as_tensor(node_mappings[lev], dtype=torch.long)]
    for lev in range(len(graphs))
]

# -----------------------------
# Build decoder and load trained weights
# -----------------------------
pooling_schedule = POOLING_SCHEDULE
unpooling_schedule = UNPOOLING_SCHEDULE

level_connections = level_connections_local[:len(pooling_schedule)]
unpool_level_connections = unpool_level_connections_local[:len(pooling_schedule)]
graphs = graphs[:len(pooling_schedule) + 1]
unpooling_schedule = unpooling_schedule[:len(level_connections)]
edge_attrs = edge_attrs[:len(pooling_schedule) + 1]
unpool_edge_attrs = list(reversed(unpool_edge_attrs[:len(pooling_schedule)]))[:len(unpooling_schedule)]
decoder_static_feats = decoder_static_feats[:len(pooling_schedule) + 1]

decoder = ProgressiveDecoder(
    latent_dim=LATENT_DIM,
    hidden_dims=DECODER_HIDDEN_DIMS,
    out_dim=OUT_DIM,
    graphs=graphs,
    level_connections=list(reversed(unpool_level_connections)),
    unpooling_schedule=unpooling_schedule,
    edge_attrs=edge_attrs,
    interlevel_edge_attrs=unpool_edge_attrs,
    static_feats=decoder_static_feats,
    heads=HEADS,
    gat_dropout=GAT_DROPOUT,
    feature_dropout=FEATURE_DROPOUT,
    decoder_mlp_ratio=DECODER_MLP_RATIO,
    use_residualsIO=True,
).to(device)

state_dict = torch.load(WEIGHTS / f"{EXP_ID}_best_model.pt", map_location=device)
state_dict = clean_state_dict(state_dict)
decoder.load_state_dict(decoder_state_dict_from_full_checkpoint(state_dict))
decoder.eval()

num_params = sum(p.numel() for p in decoder.parameters())
print(f"Decoder loaded: {num_params:,} parameters")

# -----------------------------
# Random latent -> decode
# -----------------------------
n_coarse = len(node_mappings[1])  # 8929 for v94
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

"""
print('registering buffers')
for (name, tensor) in {
    'in_x': z,
    'out_forward': decoded,
    }.items():
    decoder.register_buffer(name, tensor)

filename = 'scripted_model.pt'
print(f'saving scripted model into "{filename}"')
scripted_model = torch.jit.script(decoder)
scripted_model.save(filename)
"""
