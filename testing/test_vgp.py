from collections import namedtuple
import pytest
import tensorflow as tf
import numpy as np
import gpflow
from VFF.vgp import VGP_1d, VGP_additive, VGP_kron, VGP_kron_anyvar

Setup = namedtuple("Setup", ["a", "b", "ms", "data", "Xt_int", "Xt_ext"])


@pytest.fixture
def setup_1d():
    N = 13
    Nt = 7
    ms = np.arange(5)
    a = np.float64(-0.3)
    b = np.float64(1.3)
    X = np.random.rand(N, 1)
    Y = np.random.randn(N, 1)
    Xt_int = np.random.rand(Nt, 1)
    Xt_ext = 10 * np.random.rand(Nt, 1) - 5
    return Setup(a=a, b=b, ms=ms, data=(X, Y), Xt_int=Xt_int, Xt_ext=Xt_ext)


@pytest.fixture
def setup_2d(setup_1d):
    N = 13
    Nt = 7
    ms = np.arange(5)
    a = np.array([-0.3, -0.2])
    b = np.array([1.3, 1.5])
    X = np.random.rand(N, 2)
    Y = np.random.randn(N, 1)
    Xt_int = np.random.rand(Nt, 2)
    Xt_ext = 10 * np.random.rand(Nt, 2) - 5
    return Setup(a=a, b=b, ms=ms, data=(X, Y), Xt_int=Xt_int, Xt_ext=Xt_ext)


def optimize(m):
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=5))


def predict(m, Xnew):
    return m.predict_f(Xnew)


matern_kernels = [
    matern_class(lengthscales=0.7, variance=0.6)
    for matern_class in [gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52]
]
matern_kernels_not_52 = matern_kernels[:2] + [
    pytest.param(matern_kernels[2], marks=pytest.mark.xfail)
]


@pytest.mark.parametrize("kernel", matern_kernels)
@pytest.mark.parametrize("likelihood", [gpflow.likelihoods.Gaussian(0.4)])
def test_VGP_1d(setup_1d, kernel, likelihood):
    m = VGP_1d(setup_1d.data, setup_1d.ms, setup_1d.a, setup_1d.b, kernel, likelihood)
    optimize(m)
    _ = predict(m, setup_1d.Xt_int)
    if not isinstance(kernel, gpflow.kernels.Matern52):
        # not yet implemented
        _ = predict(m, setup_1d.Xt_ext)


@pytest.mark.parametrize("kernel1", matern_kernels_not_52)
@pytest.mark.parametrize("kernel2", matern_kernels_not_52)
@pytest.mark.parametrize("likelihood", [gpflow.likelihoods.Gaussian(0.4)])
def test_VGP_additive(setup_2d, kernel1, kernel2, likelihood):
    kernels = [kernel1, kernel2]
    m = VGP_additive(setup_2d.data, setup_2d.ms, setup_2d.a, setup_2d.b, kernels, likelihood)
    optimize(m)
    _ = predict(m, setup_2d.Xt_int)
    if not any(isinstance(kernel, gpflow.kernels.Matern52) for kernel in kernels):
        # not yet implemented
        _ = predict(m, setup_2d.Xt_ext)


@pytest.mark.parametrize("kernel1", matern_kernels)
@pytest.mark.parametrize("kernel2", matern_kernels)
@pytest.mark.parametrize("likelihood", [gpflow.likelihoods.Gaussian(0.4)])
@pytest.mark.parametrize(
    "extra_options",
    [{}, dict(use_two_krons=True), dict(use_extra_ranks=1), dict(use_extra_ranks=2)],
)
def test_VGP_kron(setup_2d, kernel1, kernel2, likelihood, extra_options):
    kernels = [kernel1, kernel2]
    m = VGP_kron(
        setup_2d.data, setup_2d.ms, setup_2d.a, setup_2d.b, kernels, likelihood, **extra_options
    )
    optimize(m)
    _ = predict(m, setup_2d.Xt_int)
    if not any(isinstance(kernel, gpflow.kernels.Matern52) for kernel in kernels):
        # not yet implemented
        _ = predict(m, setup_2d.Xt_ext)


@pytest.mark.parametrize("kernel1", matern_kernels)
@pytest.mark.parametrize("kernel2", matern_kernels)
@pytest.mark.parametrize("likelihood", [gpflow.likelihoods.Gaussian(0.4)])
def test_VGP_kron_anyvar(setup_2d, kernel1, kernel2, likelihood):
    kernels = [kernel1, kernel2]
    m = VGP_kron_anyvar(setup_2d.data, setup_2d.ms, setup_2d.a, setup_2d.b, kernels, likelihood)
    optimize(m)
    _ = predict(m, setup_2d.Xt_int)
    if not any(isinstance(kernel, gpflow.kernels.Matern52) for kernel in kernels):
        # not yet implemented
        _ = predict(m, setup_2d.Xt_ext)
