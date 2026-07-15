
from cpp_torch.modules.wrapped_module import WrappedModule


def construct_module(name):
    match name:
        case 'small-mlp':
            from cpp_torch.modules.multi_layer_perceptron import SmallMLP
            return WrappedModule(SmallMLP())
        case 'gnn-sage':
            from cpp_torch.modules.gnn_sage_conv import SmallGNN
            return WrappedModule(SmallGNN())
        case 'gnn-gatv2':
            from cpp_torch.modules.gnn_gatv2 import SmallGNN
            return WrappedModule(SmallGNN())
        case 'gnn-gatv3':
            from cpp_torch.modules.gnn_gatv3 import SmallGNN
            return WrappedModule(SmallGNN())
        case 'zigas-decoder':
            from pathlib import Path
            import numpy
            import torch
            from cpp_torch.modules.models14 import ZigasDecoder
            torch.serialization.add_safe_globals([numpy.dtype])
            torch.serialization.add_safe_globals([numpy.ndarray])
            torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            BASE = Path(__file__).resolve().parent
            EXP_ID = "v94"
            N_SUBGRAPHS = 6
            OUT_DIM = 114
            ENCODER_HIDDEN_DIMS = [256, 256, 192, 128, 96]
            DECODER_HIDDEN_DIMS = ENCODER_HIDDEN_DIMS[::-1]
            LATENT_DIM = 64
            HEADS = 4
            GAT_DROPOUT = 0.1
            FEATURE_DROPOUT = 0.1
            POOLING_SCHEDULE = [2]
            UNPOOLING_SCHEDULE = [3]
            DECODER_MLP_RATIO = 2.0
            return WrappedModule(ZigasDecoder(
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
            ))
        case _:
            raise ValueError(f'unknown module: {name}')
