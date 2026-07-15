"""Minimal working example: decode a random latent tensor with the trained
v94 ProgressiveGraphAutoencoder decoder (models13.py).

Self-contained: every file this script needs lives under test_mwe/
(graph/, weights/, data/, models13.py). Run with:

    /perm/pazz/conda/envs/torch/bin/python mwe.py
"""
from pathlib import Path

import numpy as np
import torch

from models13 import ProgressiveGraphAutoencoder

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
# Build model and load trained weights
# -----------------------------
model = ProgressiveGraphAutoencoder(
    in_dim=IN_DIM_ACTIVE,
    encoder_hidden_dims=ENCODER_HIDDEN_DIMS,
    decoder_hidden_dims=DECODER_HIDDEN_DIMS,
    latent_dim=LATENT_DIM,
    out_dim=OUT_DIM,
    graphs=graphs,
    level_connections=level_connections_local,
    unpool_level_connections=unpool_level_connections_local,
    pooling_schedule=POOLING_SCHEDULE,
    unpooling_schedule=UNPOOLING_SCHEDULE,
    edge_attrs=edge_attrs,
    pool_interlevel_edge_attrs=pool_edge_attrs,
    unpool_interlevel_edge_attrs=unpool_edge_attrs,
    decoder_static_feats=decoder_static_feats,
    heads=HEADS,
    gat_dropout=GAT_DROPOUT,
    feature_dropout=FEATURE_DROPOUT,
    latent_dropout=LATENT_DROPOUT,
    latent_noise_std=LATENT_NOISE_STD,
    encoder_mlp_ratio=ENCODER_MLP_RATIO,
    decoder_mlp_ratio=DECODER_MLP_RATIO,
    use_residualsIO=True,
    vae=USE_VAE,
    free_bits=FREE_BITS,
).to(device)

state_dict = torch.load(WEIGHTS / f"{EXP_ID}_best_model.pt", map_location=device)
model.load_state_dict(clean_state_dict(state_dict))
model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {num_params:,} parameters")

# -----------------------------
# Random latent -> decode
# -----------------------------
n_coarse = len(node_mappings[1])  # 8929 for v94
z = torch.randn(n_coarse, LATENT_DIM, device=device)
print(f"Random latent shape: {tuple(z.shape)}")

with torch.no_grad():
    decoded = model.decode(z)

print(f"Decoded output shape: {tuple(decoded.shape)}")
print(f"decoded stats -> mean: {decoded.mean().item():.4f}, "
        f"std: {decoded.std().item():.4f}, "
        f"min: {decoded.min().item():.4f}, max: {decoded.max().item():.4f}")

out_path = BASE / "output/decoded_output_13.pt"
torch.save(decoded.cpu(), out_path)
print(f"Saved decoded output to {out_path}")

