import logging
import tomllib

import numpy as np
import torch

from cpp_torch.wdir import wdir
from cpp_torch.modules import construct_module
from cpp_torch.modules.wrapped_module import WrappedModule
from cpp_torch.io import load_tensor
from cpp_torch.compare import compare_tensors, compare_arrays

logger = logging.getLogger(__name__)


def check():
    filename = wdir / 'config.toml'
    logger.info(f'reading config from "{filename}"')
    with open(filename, 'rb') as f:
        config = tomllib.load(f)

    logger.info('creating model')
    model = construct_module(**config)
    wrapped_model = WrappedModule(model)

    logger.info('loading data')
    p = load_tensor(wdir / 'p.bin')
    x = load_tensor(wdir / 'x.bin')
    dy = load_tensor(wdir / 'dy.bin')
    dx = load_tensor(wdir / 'dx.bin')
    dp = load_tensor(wdir / 'dp.bin')
    dx_0 = torch.zeros_like(dx)
    dp_0 = torch.zeros_like(dp)
    FpT_dy_cpp = load_tensor(wdir / 'FpT_dy.bin')
    FxT_dy_cpp = load_tensor(wdir / 'FxT_dy.bin')
    Fx_dx_cpp = load_tensor(wdir / 'Fx_dx.bin')
    Fp_dp_cpp = load_tensor(wdir / 'Fp_dp.bin')
    y_cpp = load_tensor(wdir / 'y.bin')

    logger.info('applying forward')
    y_py = wrapped_model.forward(p, x)
    compare_tensors('forward', y_cpp, y_py, rtol=1e-5, atol=0)

    logger.info('applying adjoint')
    ad_py = wrapped_model.apply_ad(dy)
    FpT_dy_py = ad_py[0]
    FxT_dy_py = ad_py[1]
    compare_tensors('FpT_dy', FpT_dy_cpp, FpT_dy_py, rtol=1e-4, atol=0)
    compare_tensors('FxT_dy', FxT_dy_cpp, FxT_dy_py, rtol=1e-5, atol=0)

    logger.info('applying tangent linear')
    Fx_dx_py = wrapped_model.apply_tl(dp_0, dx)
    Fp_dp_py = wrapped_model.apply_tl(dp, dx_0)
    compare_tensors('Fp_dp', Fp_dp_cpp, Fp_dp_py, rtol=1e-5, atol=0)
    compare_tensors('Fx_dx', Fx_dx_cpp, Fx_dx_py, rtol=1e-5, atol=0)

    logger.info('computing adjoint test')
    dot_1 = np.array([FpT_dy_cpp @ dp + FxT_dy_cpp @ dx])
    dot_2 = np.array([(Fx_dx_cpp + Fp_dp_cpp) @ dy])
    compare_arrays('dot', dot_1, dot_2, rtol=1e-5, atol=0, n_first=1)
