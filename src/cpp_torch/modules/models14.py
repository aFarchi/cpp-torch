import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import TransformerConv, LayerNorm
from torch_geometric.utils import softmax as pyg_softmax
from torch.utils.checkpoint import checkpoint
import re
from pathlib import Path
from typing import List, Dict, Optional

EDGE_DIM = 4   # [dlat, sin(dlon_wrap), cos(dlon_wrap), log(haversine+eps)]


# ===========================================================================
# Attention-based pooling / unpooling  (Change 4)
# ===========================================================================

class AttentionPooling(nn.Module):
    """Fine → coarse aggregation with geometry-conditioned attention scores."""
    def __init__(self, connections_dict: Dict[int, List[int]],
                 in_dim: int, num_heads: int = 4,
                 edge_attr: Optional[torch.Tensor] = None):
        super().__init__()
        assert in_dim % num_heads == 0
        self.in_dim    = in_dim
        self.num_heads = num_heads
        self.head_dim  = in_dim // num_heads
        self.edge_dim  = edge_attr.shape[-1] if edge_attr is not None else 0

        coarse_keys = sorted(connections_dict.keys())
        self.n_coarse = len(coarse_keys)
        c2l = {g: l for l, g in enumerate(coarse_keys)}

        fine_srcs, coarse_dsts, max_fine = [], [], 0
        for ck, fl in connections_dict.items():
            cl = c2l[ck]
            for fi in fl:
                fine_srcs.append(fi)
                coarse_dsts.append(cl)
                if fi > max_fine:
                    max_fine = fi

        self.n_fine = max_fine + 1
        self.register_buffer('edge_src',        torch.tensor(fine_srcs,   dtype=torch.long))
        self.register_buffer('edge_dst',        torch.tensor(coarse_dsts, dtype=torch.long))
        self.register_buffer('inter_edge_attr', edge_attr)

        self.value_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.score_proj = nn.Linear(in_dim + self.edge_dim, num_heads, bias=False)
        self.out_proj   = nn.Linear(in_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[N_fine, C] → [N_coarse, C]"""
        values_e = self.value_proj(x)[self.edge_src]

        feat_e = x[self.edge_src]
        if self.inter_edge_attr is not None:
            score_input = torch.cat([feat_e, self.inter_edge_attr], dim=-1)
        else:
            score_input = feat_e
        scores_e = self.score_proj(score_input)

        attn     = pyg_softmax(scores_e, self.edge_dst, num_nodes=self.n_coarse)
        values_e = values_e.view(-1, self.num_heads, self.head_dim)
        weighted = (attn.unsqueeze(-1) * values_e).view(-1, self.in_dim)

        out = torch.zeros(self.n_coarse, self.in_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, self.edge_dst, weighted)
        return self.out_proj(out)


class AttentionUnpooling(nn.Module):
    """Coarse → fine interpolation with per-edge attention and FiLM geometry.

    models13 change: models12 summed parent contributions unnormalised
    (index_add_ with no weights, num_heads unused), so fine-node magnitude
    depended on parent count and all children of a coarse node received the
    same vector.  Here per-edge scores from (coarse feature, edge geometry)
    are softmax-normalised over each fine node's coarse parents, mirroring
    AttentionPooling."""
    def __init__(self, connections_dict: Dict[int, List[int]],
                 in_dim: int, num_heads: int = 4,
                 edge_attr: Optional[torch.Tensor] = None):
        super().__init__()
        assert in_dim % num_heads == 0
        self.in_dim    = in_dim
        self.num_heads = num_heads
        self.head_dim  = in_dim // num_heads
        self.edge_dim  = edge_attr.shape[-1] if edge_attr is not None else 0

        coarse_keys = sorted(connections_dict.keys())
        self.n_coarse = len(coarse_keys)
        c2l = {g: l for l, g in enumerate(coarse_keys)}

        coarse_srcs, fine_dsts, max_fine = [], [], 0
        for ck, fl in connections_dict.items():
            cl = c2l[ck]
            for fi in fl:
                coarse_srcs.append(cl)
                fine_dsts.append(fi)
                if fi > max_fine:
                    max_fine = fi

        self.n_fine = max_fine + 1
        self.register_buffer('edge_src',        torch.tensor(coarse_srcs, dtype=torch.long))
        self.register_buffer('edge_dst',        torch.tensor(fine_dsts,   dtype=torch.long))
        self.register_buffer('inter_edge_attr', edge_attr)

        self.value_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.score_proj = nn.Linear(in_dim + self.edge_dim, num_heads, bias=False)
        if self.edge_dim > 0:
            self.geo_proj = nn.Linear(self.edge_dim, in_dim * 2)
            nn.init.zeros_(self.geo_proj.weight)
            nn.init.zeros_(self.geo_proj.bias)
        self.out_proj = nn.Linear(in_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[N_coarse, C] → [N_fine, C]"""
        coarse_e = x[self.edge_src]
        values_e = self.value_proj(coarse_e)

        if self.inter_edge_attr is not None:
            film = self.geo_proj(self.inter_edge_attr)
            scale, shift = film.chunk(2, dim=-1)
            values_e = values_e * (1 + scale) + shift
            score_input = torch.cat([coarse_e, self.inter_edge_attr], dim=-1)
        else:
            score_input = coarse_e
        scores_e = self.score_proj(score_input)

        attn     = pyg_softmax(scores_e, self.edge_dst, num_nodes=self.n_fine)
        values_e = values_e.view(-1, self.num_heads, self.head_dim)
        weighted = (attn.unsqueeze(-1) * values_e).view(-1, self.in_dim)

        out = torch.zeros(self.n_fine, self.in_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, self.edge_dst, weighted)
        return self.out_proj(out)


# ===========================================================================
# Pre-norm TransformerConv block with MLP sub-block
#
# Replaces GATv2Conv (additive attention) with TransformerConv (scaled
# dot-product attention).  Edge features enter the key computation:
#   alpha_ij ∝ (W_Q h_i)^T (W_K h_j + W_E e_ij) / sqrt(d_k)
# which gives direction- and distance-aware attention without additive mixing.
#
# root_weight=False: the skip_proj residual in GATBlock handles the
# self-connection, avoiding an unintended double-skip if root_weight=True.
# ===========================================================================

class GATBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 heads: int, gat_dropout: float, feature_dropout: float,
                 mlp_ratio: float = 1.0, edge_dim: int = EDGE_DIM):
        super().__init__()
        assert out_dim % heads == 0
        self.in_dim   = in_dim
        self.out_dim  = out_dim
        self.edge_dim = edge_dim

        # Attention sub-block (pre-norm)
        self.norm1       = LayerNorm(in_dim)
        self.transformer = TransformerConv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            concat=True,
            dropout=gat_dropout,
            edge_dim=edge_dim if edge_dim > 0 else None,
            root_weight=False,  # skip_proj below handles the residual
            beta=False,
            bias=True,
        )
        self.skip_proj = (nn.Linear(in_dim, out_dim, bias=False)
                          if in_dim != out_dim else nn.Identity())

        # MLP sub-block (pre-norm)
        self.norm2  = LayerNorm(out_dim)
        mlp_dim     = int(out_dim * mlp_ratio)
        self.mlp    = nn.Sequential(
            nn.Linear(out_dim, mlp_dim),
            nn.GELU(approximate='tanh'),
            nn.Dropout(feature_dropout),
            nn.Linear(mlp_dim, out_dim),
            nn.Dropout(feature_dropout),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        ea = edge_attr if self.edge_dim > 0 else None
        x = self.skip_proj(x) + self.transformer(self.norm1(x), edge_index, ea)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_linear(self, x: torch.Tensor, edge_index: torch.Tensor,
                       edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Approximate linearised forward: MLP sub-block removed.
        Use with torch.func.jvp/vjp in the DA inner loop."""
        ea = edge_attr if self.edge_dim > 0 else None
        x = self.skip_proj(x) + self.transformer(self.norm1(x), edge_index, ea)
        return x


# ===========================================================================
# Residual helper
# ===========================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = (nn.Linear(in_dim, out_dim)
                           if in_dim != out_dim else nn.Identity())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
        return x_out + self.dropout(self.projection(x_in))


class ResidualOutBlock(nn.Module):
    """Output projection with optional residual branch."""
    def __init__(self, in_dim: int, out_dim: int,
                 feature_dropout: float, use_residualsIO: bool):
        super().__init__()
        self.output_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(in_dim, out_dim),
        )
        self.output_residual = (ResidualBlock(in_dim, out_dim, feature_dropout * 0.5)
                                if use_residualsIO else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.output_proj(x)
        if self.output_residual is not None:
            x = self.output_residual(x_in, x)
        return x


class GraphGATBlock(nn.Module):
    """GATBlock bound to fixed graph connectivity and edge attributes."""
    def __init__(self, gat_block: GATBlock,
                 edge_index: torch.Tensor,
                 edge_attr: Optional[torch.Tensor] = None):
        super().__init__()
        self.gat_block = gat_block
        # Structural tensors come from config data and should not participate
        # in checkpoint key matching.
        self.register_buffer('edge_index', edge_index, persistent=False)
        self.register_buffer('edge_attr', edge_attr, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gat_block(x, self.edge_index, self.edge_attr)


class UnpoolAndStaticFilmBlock(nn.Module):
    """Unpool stage with optional static-field FiLM conditioning."""
    def __init__(self, unpooler: AttentionUnpooling,
                 static_feat: Optional[torch.Tensor], in_dim: int):
        super().__init__()
        self.unpooler = unpooler
        self.has_static = static_feat is not None
        if self.has_static:
            # Static fields are deterministic inputs, not learned state.
            self.register_buffer('static_feat', static_feat, persistent=False)
            self.film = nn.Linear(static_feat.shape[-1], in_dim * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unpooler(x)
        if self.has_static:
            scale, shift = self.film(self.static_feat).chunk(2, dim=-1)
            x = x * (1 + scale) + shift
        return x


class InputProjResidualBlock(nn.Module):
    """Encoder input projection + norm/activation/dropout + optional residual."""
    def __init__(self, in_dim: int, out_dim: int,
                 feature_dropout: float, use_residualsIO: bool):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, out_dim)
        self.input_norm = LayerNorm(out_dim)
        self.feature_dropout = feature_dropout
        self.input_residual = (ResidualBlock(in_dim, out_dim, feature_dropout * 0.5)
                               if use_residualsIO else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = F.gelu(self.input_norm(self.input_proj(x)), approximate='tanh')
        x = F.dropout(x, p=self.feature_dropout, training=self.training)
        if self.input_residual is not None:
            x = self.input_residual(x_in, x)
        return x


class AttentionPoolBlock(nn.Module):
    """Wrap AttentionPooling as a simple Sequential stage."""
    def __init__(self, pooler: AttentionPooling):
        super().__init__()
        self.pooler = pooler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooler(x)


class VAELatentHead(nn.Module):
    """VAE latent projection head producing (z, mu, logvar)."""
    def __init__(self, in_dim: int, latent_dim: int, feature_dropout: float):
        super().__init__()
        mid = max(latent_dim * 2, in_dim)
        self.latent_trunk = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.GELU(approximate='tanh'),
            nn.Dropout(feature_dropout),
        )
        self.mu_head = nn.Linear(mid, latent_dim)
        self.logvar_head = nn.Linear(mid, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.latent_trunk(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-30., 20.)
        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu
        return z, mu, logvar


class AELatentHead(nn.Module):
    """AE latent projection head producing (z, None, None)."""
    def __init__(self, in_dim: int, latent_dim: int,
                 feature_dropout: float,
                 latent_dropout: float,
                 latent_noise_std: float):
        super().__init__()
        mid = max(latent_dim * 2, in_dim)
        self.latent_proj = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.GELU(approximate='tanh'),
            nn.Dropout(feature_dropout),
            nn.Linear(mid, latent_dim),
        )
        self.latent_dropout = latent_dropout
        self.latent_noise_std = latent_noise_std

    def forward(self, x: torch.Tensor):
        x = self.latent_proj(x)
        if self.training:
            if self.latent_dropout > 0:
                x = F.dropout(x, p=self.latent_dropout, training=True)
            if self.latent_noise_std > 0:
                x = x + torch.randn_like(x) * self.latent_noise_std
        return x, None, None


# ===========================================================================
# Progressive Encoder
# ===========================================================================

class ProgressiveEncoder(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dims: List[int], latent_dim: int,
                 graphs: List[torch.Tensor],
                 level_connections: List[Dict[int, List[int]]],
                 pooling_schedule: List[int],
                 edge_attrs: Optional[List[torch.Tensor]] = None,
                 interlevel_edge_attrs: Optional[List[torch.Tensor]] = None,
                 heads: int = 4, gat_dropout: float = 0.2,
                 feature_dropout: float = 0.3, latent_dropout: float = 0.1,
                 latent_noise_std: float = 0.05,
                 encoder_mlp_ratio: float = 1.0,
                 use_residualsIO: bool = False,
                 vae: bool = False,
                 log: bool = False):
        super().__init__()
        self.graphs            = graphs
        self.level_connections = level_connections
        if edge_attrs is not None:
            for i, ea in enumerate(edge_attrs):
                self.register_buffer(f'edge_attr_{i}', ea)
            self._n_edge_attrs = len(edge_attrs)
        else:
            self._n_edge_attrs = 0
        self.pooling_schedule  = sorted(pooling_schedule)
        self.feature_dropout   = feature_dropout
        self.latent_dropout    = latent_dropout
        self.latent_noise_std  = latent_noise_std
        self.vae               = vae

        all_dims = [in_dim] + hidden_dims + [latent_dim]
        edge_dim = EDGE_DIM if edge_attrs is not None else 0

        input_block = InputProjResidualBlock(
            in_dim=in_dim,
            out_dim=all_dims[1],
            feature_dropout=feature_dropout,
            use_residualsIO=use_residualsIO,
        )
        self.add_module('input_block', input_block)

        gat_blocks = nn.ModuleList([
            GATBlock(all_dims[i], all_dims[i + 1], heads,
                     gat_dropout, feature_dropout,
                     mlp_ratio=encoder_mlp_ratio, edge_dim=edge_dim)
            for i in range(1, len(all_dims) - 2)
        ])

        poolers = nn.ModuleDict()
        for pool_step, layer_idx in enumerate(self.pooling_schedule):
            out_f   = all_dims[layer_idx + 1]
            pool_ea = (interlevel_edge_attrs[pool_step]
                       if interlevel_edge_attrs is not None else None)
            poolers[str(layer_idx)] = AttentionPooling(
                level_connections[pool_step], in_dim=out_f, num_heads=heads,
                edge_attr=pool_ea)

        # Build encoder stages in forward execution order, equivalent to
        # the previous non-checkpointed path.
        graph_level = 0
        if 0 in self.pooling_schedule:
            self.add_module('pool_stage_0', AttentionPoolBlock(poolers['0']))
            graph_level += 1

        for i, blk in enumerate(gat_blocks):
            layer_idx = i + 1
            ea = getattr(self, f'edge_attr_{graph_level}') if self._n_edge_attrs > 0 else None
            self.add_module(
                f'gat_stage_{layer_idx}',
                GraphGATBlock(gat_block=blk,
                              edge_index=self.graphs[graph_level],
                              edge_attr=ea),
            )

            if layer_idx in self.pooling_schedule:
                self.add_module(
                    f'pool_stage_{layer_idx}',
                    AttentionPoolBlock(poolers[str(layer_idx)]),
                )
                graph_level += 1

        if vae:
            self.add_module('latent_head', VAELatentHead(
                in_dim=all_dims[-2],
                latent_dim=latent_dim,
                feature_dropout=feature_dropout,
            ))
        else:
            self.add_module('latent_head', AELatentHead(
                in_dim=all_dims[-2],
                latent_dim=latent_dim,
                feature_dropout=feature_dropout,
                latent_dropout=latent_dropout,
                latent_noise_std=latent_noise_std,
            ))


# ===========================================================================
# Progressive Decoder
# ===========================================================================

class ProgressiveDecoder(nn.Sequential):
    def __init__(self, latent_dim: int, hidden_dims: List[int], out_dim: int,
                 graphs: List[torch.Tensor],
                 level_connections: List[Dict[int, List[int]]],
                 unpooling_schedule: List[int],
                 edge_attrs: Optional[List[torch.Tensor]] = None,
                 interlevel_edge_attrs: Optional[List[torch.Tensor]] = None,
                 static_feats: Optional[List[Optional[torch.Tensor]]] = None,
                 heads: int = 4, gat_dropout: float = 0.2,
                 feature_dropout: float = 0.3,
                 decoder_mlp_ratio: float = 1.0,
                 use_residualsIO: bool = False,
                 log: bool = False):
        super().__init__()
        self.graphs             = graphs
        self.level_connections  = level_connections
        if edge_attrs is not None:
            for i, ea in enumerate(edge_attrs):
                self.register_buffer(f'edge_attr_{i}', ea)
            self._n_edge_attrs = len(edge_attrs)
        else:
            self._n_edge_attrs = 0
        self.unpooling_schedule = sorted(unpooling_schedule)
        self.feature_dropout    = feature_dropout

        all_dims = [latent_dim] + hidden_dims + [out_dim]
        edge_dim = EDGE_DIM if edge_attrs is not None else 0

        mid = max(latent_dim * 2, all_dims[1])
        latent_expand = nn.Sequential(
            nn.Linear(latent_dim, mid),
            nn.GELU(approximate='tanh'),
            nn.Dropout(feature_dropout),
            nn.Linear(mid, all_dims[1]),
        )
        expand_norm = LayerNorm(all_dims[1])
        self.add_module('latent_expand', latent_expand)
        self.add_module('expand_norm', expand_norm)
        self.add_module('expand_act', nn.GELU(approximate='tanh'))
        self.add_module('expand_dropout', nn.Dropout(feature_dropout))

        gat_blocks = nn.ModuleList([
            GATBlock(all_dims[i], all_dims[i + 1], heads,
                     gat_dropout, feature_dropout,
                     mlp_ratio=decoder_mlp_ratio, edge_dim=edge_dim)
            for i in range(1, len(all_dims) - 2)
        ])

        # Build decoder stages in forward execution order, equivalent to decode_nockpt.
        graph_level  = len(self.graphs) - 1
        unpool_step  = len(self.unpooling_schedule) - 1
        for i, blk in enumerate(gat_blocks):
            layer_idx = i + 1

            if layer_idx in self.unpooling_schedule and unpool_step >= 0:
                in_f = all_dims[layer_idx]
                unpool_ea = (interlevel_edge_attrs[unpool_step]
                             if interlevel_edge_attrs is not None else None)
                unpooler = AttentionUnpooling(
                    level_connections[unpool_step], in_dim=in_f, num_heads=heads,
                    edge_attr=unpool_ea)

                dest_level = len(graphs) - 2 - unpool_step
                sf = (static_feats[dest_level]
                      if static_feats is not None and dest_level < len(static_feats)
                      else None)

                self.add_module(
                    f'unpool_and_film_{layer_idx}',
                    UnpoolAndStaticFilmBlock(unpooler=unpooler,
                                             static_feat=sf,
                                             in_dim=in_f),
                )
                unpool_step -= 1
                graph_level -= 1

            ea = getattr(self, f'edge_attr_{graph_level}') if self._n_edge_attrs > 0 else None
            self.add_module(
                f'gat_stage_{layer_idx}',
                GraphGATBlock(gat_block=blk,
                              edge_index=self.graphs[graph_level],
                              edge_attr=ea),
            )

        self.add_module('output_head', ResidualOutBlock(
            in_dim=all_dims[-2],
            out_dim=out_dim,
            feature_dropout=feature_dropout,
            use_residualsIO=use_residualsIO,
        ))


class ZigasDecoder(ProgressiveDecoder):

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        exp_id: str = 'v94',
        n_subgraphs: int = 6,
        static_vars: Optional[List[str]] = None,
        latent_dim: int = 64,
        out_dim: int = 114,
        decoder_hidden_dims: Optional[List[int]] = None,
        heads: int = 4,
        gat_dropout: float = 0.1,
        feature_dropout: float = 0.1,
        decoder_mlp_ratio: float = 2.0,
        pooling_schedule: Optional[List[int]] = None,
        unpooling_schedule: Optional[List[int]] = None,
        weights_filename: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent
        if static_vars is None:
            static_vars = ["sin_lat", "cos_lat", "sin_lon", "cos_lon", "lnsurfgeo", "lsm"]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [96, 128, 192, 256, 256]
        if pooling_schedule is None:
            pooling_schedule = [2]
        if unpooling_schedule is None:
            unpooling_schedule = [3]
        if weights_filename is None:
            weights_filename = f'{exp_id}_best_model.pt'
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        graph_dir = base_dir / 'graph'
        data_dir = base_dir / 'data'
        weights_dir = base_dir / 'weights'

        # Allow loading checkpoints that include NumPy globals.
        torch.serialization.add_safe_globals([np.dtype, np.ndarray, np._core.multiarray._reconstruct])

        graphs = [
            torch.load(graph_dir / f'{exp_id}_edge_index_ae_{lev}.pt')
            for lev in range(n_subgraphs + 1)
        ]
        edge_attrs = torch.load(graph_dir / f'{exp_id}_edge_attrs.pt')
        level_connections_local = torch.load(graph_dir / f'{exp_id}_level_connections_local.pt')
        unpool_level_connections_local = torch.load(
            graph_dir / f'{exp_id}_unpool_level_connections_local.pt')
        node_mappings = torch.load(graph_dir / f'{exp_id}_node_mappings.pt', weights_only=False)
        unpool_edge_attrs = torch.load(graph_dir / f'{exp_id}_unpool_interlevel_edge_attrs.pt')

        graphs = [g.to(device) for g in graphs]

        statics_full = torch.tensor(
            np.stack([np.load(data_dir / f'{v}.npy') for v in static_vars], axis=-1),
            dtype=torch.float32)
        decoder_static_feats = [
            statics_full[torch.as_tensor(node_mappings[lev], dtype=torch.long)]
            for lev in range(len(graphs))
        ]

        level_connections = level_connections_local[:len(pooling_schedule)]
        unpool_level_connections = unpool_level_connections_local[:len(pooling_schedule)]
        graphs = graphs[:len(pooling_schedule) + 1]
        unpooling_schedule = unpooling_schedule[:len(level_connections)]
        edge_attrs = edge_attrs[:len(pooling_schedule) + 1]
        unpool_edge_attrs = list(reversed(unpool_edge_attrs[:len(pooling_schedule)]))[:len(unpooling_schedule)]
        decoder_static_feats = decoder_static_feats[:len(pooling_schedule) + 1]

        super().__init__(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            out_dim=out_dim,
            graphs=graphs,
            level_connections=list(reversed(unpool_level_connections)),
            unpooling_schedule=unpooling_schedule,
            edge_attrs=edge_attrs,
            interlevel_edge_attrs=unpool_edge_attrs,
            static_feats=decoder_static_feats,
            heads=heads,
            gat_dropout=gat_dropout,
            feature_dropout=feature_dropout,
            decoder_mlp_ratio=decoder_mlp_ratio,
            use_residualsIO=True,
        )

        self.exp_id = exp_id
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.device_ref = device

        state_dict = torch.load(weights_dir / weights_filename, map_location=device)
        state_dict = self._clean_state_dict(state_dict)
        self.load_state_dict(self._decoder_state_dict_from_full_checkpoint(state_dict))
        self.to(device)
        self.eval()
        self.input_shape = (8929, 64)
        self.output_shape = (35718, 114)

    @staticmethod
    def _clean_state_dict(state_dict, prefixes=("_orig_mod.",)):
        cleaned = {}
        for k, v in state_dict.items():
            new_k = k
            for p in prefixes:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
            cleaned[new_k] = v
        return cleaned

    @staticmethod
    def _decoder_state_dict_from_full_checkpoint(full_state_dict):
        out = {}
        for key, value in full_state_dict.items():
            if not key.startswith("decoder."):
                continue

            k = key
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

            if re.match(r"^decoder\.static_feats_\d+$", k):
                continue

            out[k[len("decoder."):]] = value

        return out
        
# ===========================================================================
# Full autoencoder
# ===========================================================================

class ProgressiveGraphAutoencoder(nn.Module):
    """Progressive graph autoencoder with TransformerConv attention blocks.

    Identical to models12.ProgressiveGraphAutoencoder except:
      * AttentionUnpooling performs real attention: per-edge scores
        softmax-normalised over each fine node's coarse parents (models12
        summed parent contributions unnormalised).
      * Optional decoder_static_feats: per-level static fields (orography,
        land-sea mask, position) injected via zero-init FiLM right after each
        unpool, so the decoder can synthesise fine-scale detail.  DA-safe:
        statics are constant buffers, not part of the latent state, and the
        FiLM is affine in x.
            * Exact full-decoder TL/adjoint for DA: decode_tl (forward-mode jvp via
                the checkpoint-free decode_nockpt) and decode_adjoint (reverse-mode
                through the decoder forward).
    """
    def __init__(self, in_dim: int, latent_dim: int = 2, out_dim: int = 3,
                 encoder_hidden_dims: List[int] = None,
                 decoder_hidden_dims: List[int] = None,
                 graphs: List[torch.Tensor] = None,
                 level_connections: List[Dict[int, List[int]]] = None,
                 unpool_level_connections: Optional[List[Dict[int, List[int]]]] = None,
                 pooling_schedule: List[int] = None,
                 unpooling_schedule: List[int] = None,
                 edge_attrs: Optional[List[torch.Tensor]] = None,
                 pool_interlevel_edge_attrs: Optional[List[torch.Tensor]] = None,
                 unpool_interlevel_edge_attrs: Optional[List[torch.Tensor]] = None,
                 decoder_static_feats: Optional[List[Optional[torch.Tensor]]] = None,
                 heads: int = 4, gat_dropout: float = 0.2,
                 feature_dropout: float = 0.3,
                 latent_dropout: float = 0.1, latent_noise_std: float = 0.05,
                 encoder_mlp_ratio: float = 1.0,
                 decoder_mlp_ratio: float = 1.0,
                 use_residualsIO: bool = False,
                 vae: bool = False,
                 free_bits: float = 0.0,
                 log: bool = False):
        super().__init__()

        if graphs is None:
            raise ValueError("graphs must be provided")
        if level_connections is None:
            raise ValueError("level_connections must be provided")

        if encoder_hidden_dims is None:
            encoder_hidden_dims = [min(512, in_dim * 2), 256, 128, 64]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = list(reversed(encoder_hidden_dims))
        if pooling_schedule is None:
            pooling_schedule = list(range(2, len(encoder_hidden_dims), 2))
        if unpooling_schedule is None:
            unpooling_schedule = list(range(1, len(decoder_hidden_dims), 2))

        encoder_hidden_dims = [max(d, heads) for d in encoder_hidden_dims]
        decoder_hidden_dims = [max(d, heads) for d in decoder_hidden_dims]

        level_connections  = level_connections[:len(pooling_schedule)]
        # Decoder-side unpool graph: defaults to the encoder partition (back-compat);
        # when given (e.g. v91 multi-parent), the decoder unpoolers use it instead.
        if unpool_level_connections is None:
            unpool_level_connections = level_connections
        else:
            unpool_level_connections = unpool_level_connections[:len(pooling_schedule)]
        graphs             = graphs[:len(pooling_schedule) + 1]
        unpooling_schedule = unpooling_schedule[:len(level_connections)]
        if edge_attrs is not None:
            edge_attrs = edge_attrs[:len(pooling_schedule) + 1]
        if pool_interlevel_edge_attrs is not None:
            pool_interlevel_edge_attrs = pool_interlevel_edge_attrs[:len(pooling_schedule)]
        if unpool_interlevel_edge_attrs is not None:
            unpool_interlevel_edge_attrs = list(
                reversed(unpool_interlevel_edge_attrs[:len(pooling_schedule)])
            )[:len(unpooling_schedule)]
        if decoder_static_feats is not None:
            decoder_static_feats = decoder_static_feats[:len(pooling_schedule) + 1]

        if len(pooling_schedule) != len(level_connections):
            raise ValueError(
                f"pooling_schedule length ({len(pooling_schedule)}) must match "
                f"level_connections length ({len(level_connections)})")
        if len(unpool_level_connections) != len(level_connections):
            raise ValueError(
                f"unpool_level_connections length ({len(unpool_level_connections)}) "
                f"must match level_connections length ({len(level_connections)})")

        self.vae        = vae
        self.free_bits  = free_bits
        self.kl_per_dim = None

        self.encoder = ProgressiveEncoder(
            in_dim=in_dim, hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim, graphs=graphs,
            level_connections=level_connections,
            pooling_schedule=pooling_schedule,
            edge_attrs=edge_attrs,
            interlevel_edge_attrs=pool_interlevel_edge_attrs,
            heads=heads, gat_dropout=gat_dropout,
            feature_dropout=feature_dropout,
            latent_dropout=latent_dropout,
            latent_noise_std=latent_noise_std,
            encoder_mlp_ratio=encoder_mlp_ratio,
            use_residualsIO=use_residualsIO,
            vae=vae, log=log)

        self.decoder = ProgressiveDecoder(
            latent_dim=latent_dim, hidden_dims=decoder_hidden_dims,
            out_dim=out_dim, graphs=graphs,
            level_connections=list(reversed(unpool_level_connections)),
            unpooling_schedule=unpooling_schedule,
            edge_attrs=edge_attrs,
            interlevel_edge_attrs=unpool_interlevel_edge_attrs,
            static_feats=decoder_static_feats,
            heads=heads, gat_dropout=gat_dropout,
            feature_dropout=feature_dropout,
            decoder_mlp_ratio=decoder_mlp_ratio,
            use_residualsIO=use_residualsIO, log=log)

    @staticmethod
    def _remap_legacy_key(key: str) -> Optional[str]:
        """Map models13 keys to models14 key layout (encoder + decoder)."""
        # -----------------
        # Encoder remapping
        # -----------------
        if key.startswith('encoder.input_proj.'):
            return key.replace('encoder.input_proj.', 'encoder.input_block.input_proj.', 1)

        if key.startswith('encoder.input_norm.'):
            return key.replace('encoder.input_norm.', 'encoder.input_block.input_norm.', 1)

        if key.startswith('encoder.input_residual.'):
            return key.replace('encoder.input_residual.', 'encoder.input_block.input_residual.', 1)

        if key.startswith('encoder.gat_blocks.'):
            m = re.match(r'^encoder\.gat_blocks\.(\d+)\.(.+)$', key)
            if m is not None:
                idx = int(m.group(1)) + 1
                return f'encoder.gat_stage_{idx}.gat_block.{m.group(2)}'

        if key.startswith('encoder.poolers.'):
            m = re.match(r'^encoder\.poolers\.(\d+)\.(.+)$', key)
            if m is not None:
                layer_idx = m.group(1)
                return f'encoder.pool_stage_{layer_idx}.pooler.{m.group(2)}'

        if key.startswith('encoder.latent_trunk.'):
            return key.replace('encoder.latent_trunk.', 'encoder.latent_head.latent_trunk.', 1)

        if key.startswith('encoder.mu_head.'):
            return key.replace('encoder.mu_head.', 'encoder.latent_head.mu_head.', 1)

        if key.startswith('encoder.logvar_head.'):
            return key.replace('encoder.logvar_head.', 'encoder.latent_head.logvar_head.', 1)

        if key.startswith('encoder.latent_proj.'):
            return key.replace('encoder.latent_proj.', 'encoder.latent_head.latent_proj.', 1)

        # -----------------
        # Decoder remapping
        # -----------------
        if key.startswith('decoder.gat_blocks.'):
            m = re.match(r'^decoder\.gat_blocks\.(\d+)\.(.+)$', key)
            if m is not None:
                idx = int(m.group(1)) + 1
                return f'decoder.gat_stage_{idx}.gat_block.{m.group(2)}'

        if key.startswith('decoder.output_proj.'):
            return key.replace('decoder.output_proj.', 'decoder.output_head.output_proj.', 1)

        if key.startswith('decoder.output_residual.'):
            return key.replace('decoder.output_residual.', 'decoder.output_head.output_residual.', 1)

        if key.startswith('decoder.unpoolers.'):
            m = re.match(r'^decoder\.unpoolers\.(\d+)\.(.+)$', key)
            if m is not None:
                layer_idx = m.group(1)
                return f'decoder.unpool_and_film_{layer_idx}.unpooler.{m.group(2)}'

        if key.startswith('decoder.static_films.'):
            m = re.match(r'^decoder\.static_films\.(\d+)\.(.+)$', key)
            if m is not None:
                layer_idx = m.group(1)
                return f'decoder.unpool_and_film_{layer_idx}.film.{m.group(2)}'

        # models13 persisted static feature buffers as decoder.static_feats_{layer}
        # while models14 treats static features as non-persistent structural data.
        if re.match(r'^decoder\.static_feats_\d+$', key):
            return None

        return key

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        remapped = {}
        for key, value in state_dict.items():
            new_key = self._remap_legacy_key(key)
            if new_key is not None:
                remapped[new_key] = value
        return super().load_state_dict(remapped, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z, mu, logvar = self.encoder(x)

        kl_loss = None
        if mu is not None:   # VAE mode
            kl_elem = -0.5 * (1. + logvar - mu.pow(2) - logvar.exp())
            kl_dim  = kl_elem.mean(0)
            kl_dim_eff = torch.clamp(kl_dim, min=self.free_bits) if self.free_bits > 0.0 else kl_dim
            kl_loss = kl_dim_eff.mean()
            self.kl_per_dim = kl_dim.detach()
        else:
            self.kl_per_dim = None

        reconstructed = self.decoder(z)
        if return_latent:
            return reconstructed, z, kl_loss
        return reconstructed, kl_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns mu (VAE) or z (AE) — deterministic, suitable for DA."""
        z, mu, _ = self.encoder(x)
        return mu if mu is not None else z

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    # ------------------------------------------------------------------
    # Exact TL/adjoint of the FULL decoder — preferred for DA.
    # Call model.eval() first so dropout is off.
    # ------------------------------------------------------------------

    def decode_nockpt(self, latent: torch.Tensor) -> torch.Tensor:
        """Full decode without activation checkpointing (jvp-compatible)."""
        return self.decoder.decode_nockpt(latent)

    def decode_tl(self, z_b: torch.Tensor, dz: torch.Tensor):
        """Exact tangent-linear of the full decoder: returns (D(z_b), J dz).

        Forward-mode AD — no tape, ~2x inference memory, but cannot use
        checkpointing.  If it does not fit (large grids), use central finite
        differences of self.decode instead."""
        return torch.func.jvp(self.decoder.decode_nockpt, (z_b,), (dz,))

    def decode_adjoint(self, z_b: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """Exact adjoint of the full decoder: returns J^T lam.

        Memory-safe: reverse-mode through the decoder forward.
        For many lam at fixed z_b (4D-Var inner loop), amortise the forward:
            z = z_b.detach().requires_grad_()
            y = model.decode(z)                       # checkpointed forward
            for lam in ...:
                JT_lam = torch.autograd.grad(y, z, lam, retain_graph=True)[0]
        """
        z = z_b.detach().requires_grad_()
        y = self.decoder(z)
        return torch.autograd.grad(y, z, grad_outputs=lam)[0]
