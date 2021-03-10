import pytest
import tensorflow as tf
import numpy as np
import gpflow
from VFF.vgp import VGP_1d, VGP_additive, VGP_kron, VGP_kron_anyvar


@pytest.fixture
def data_1d():
    N = 13
    X = np.random.rand(N, 1)
    Y = np.random.randn(N, 1)
    return (X, Y)


@pytest.fixture
def data_2d():
    N = 13
    X = np.random.rand(N, 2)
    Y = np.random.randn(N, 1)
    return (X, Y)


def optimize(m):
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables)


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
def test_VGP_1d(data_1d, kernel, likelihood):
    ms = np.arange(7)
    a = np.float64(-0.3)
    b = np.float64(1.3)
    m = VGP_1d(data_1d, ms, a, b, kernel, likelihood)
    optimize(m)
    _ = predict(m, data_1d[0][:7] * 0.5)


@pytest.mark.parametrize("kernel1", matern_kernels_not_52)
@pytest.mark.parametrize("kernel2", matern_kernels_not_52)
@pytest.mark.parametrize("likelihood", [gpflow.likelihoods.Gaussian(0.4)])
def test_VGP_additive(data_2d, kernel1, kernel2, likelihood):
    ms = np.arange(7)
    a = np.array([-0.3, -0.2])
    b = np.array([1.3, 1.5])
    kernels = [kernel1, kernel2]
    m = VGP_additive(data_2d, ms, a, b, kernels, likelihood)
    optimize(m)
    _ = predict(m, data_2d[0][:7] * 0.5)


@pytest.mark.parametrize("kernel1", matern_kernels)
@pytest.mark.parametrize("kernel2", matern_kernels)
@pytest.mark.parametrize("likelihood", [gpflow.likelihoods.Gaussian(0.4)])
@pytest.mark.parametrize(
    "extra_options",
    [{}, dict(use_two_krons=True), dict(use_extra_ranks=1), dict(use_extra_ranks=2)],
)
def test_VGP_kron(data_2d, kernel1, kernel2, likelihood, extra_options):
    ms = np.arange(7)
    a = np.array([-0.3, -0.2])
    b = np.array([1.3, 1.5])
    kernels = [kernel1, kernel2]
    m = VGP_kron(data_2d, ms, a, b, kernels, likelihood, **extra_options)
    optimize(m)
    _ = predict(m, data_2d[0][:7] * 0.5)
