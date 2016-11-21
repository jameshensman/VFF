# Copyright 2016 James Hensman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from itertools import product
import numpy as np
import GPflow
import VFF
from matplotlib import pyplot as plt


def getLocations():
    return np.loadtxt('pines.csv', delimiter=',')


def getCounts(gridResolution):
    locations = np.loadtxt('pines.csv', delimiter=',')
    counts, _ = np.histogramdd(locations, bins=(gridResolution, gridResolution), range=((0, 1.), (0., 1.)))
    return counts.reshape(-1, 1)


def getGrid(gridResolution):
    linearValues = np.linspace(0., 1., gridResolution+1)
    binEdges = np.array([np.array(elem) for elem in product(linearValues, linearValues)])
    offsetValues = linearValues[:-1] + 0.5*np.diff(linearValues)[0]
    binMids = np.array([np.array(elem) for elem in product(offsetValues, offsetValues)])
    return binEdges, binMids


def set_priors(m):
    m.kerns[0].lengthscales.prior = GPflow.priors.Gamma(0.1, 1.0)
    m.kerns[1].lengthscales.prior = GPflow.priors.Gamma(0.1, 1.0)
    m.kerns[0].variance.prior = GPflow.priors.Gamma(1.0, 1.0)
    m.kerns[1].fixed = True
    m.mean_function.c.prior = GPflow.priors.Gaussian(0., 3.0)


def plot_model(m, samples, gridResolution=64):
    intensities = []
    for s in samples:
        m.set_state(s)
        mu, _ = m.predict_y(m.X.value)
        intensities.append(mu)
    intensity = np.mean(intensities, 0)
    plt.figure()
    plt.imshow(np.flipud(intensity.reshape(gridResolution, gridResolution).T),
               interpolation='nearest', extent=[0, 1, 0, 1])
    locs = getLocations()
    plt.plot(locs[:, 0], locs[:, 1], 'k.')


def build_model(M, gridResolution=64):
    binEdges, binMids = getGrid(gridResolution)
    counts = getCounts(gridResolution)
    kerns = [GPflow.kernels.Matern32(1), GPflow.kernels.Matern32(1)]
    mf = GPflow.mean_functions.Constant()
    lik = GPflow.likelihoods.Poisson()
    a, b = np.array([-1, -1]), np.array([2, 2])
    m = VFF.gpmc.GPMC_kron(binMids, counts, kerns=kerns, likelihood=lik, a=a, b=b, ms=np.arange(M), mean_function=mf)
    return m


def init_model(m):
    m.kerns[0].lengthscales = 0.1
    m.kerns[1].lengthscales = 0.1
    m.kerns.fixed = True
    m.optimize(maxiter=100)
    m.kerns.fixed = False

if __name__ == '__main__':
    Ms = [14, 16, 18, 20, 22, 24, 26, 28, 30]
    for M in Ms:
        print('M={}'.format(M))
        m = build_model(M)
        set_priors(m)
        init_model(m)
        burn = m.sample(200, epsilon=0.05, Lmin=8, Lmax=10)
        m.set_state(burn[-1])
        samples = m.sample(1000, epsilon=0.1, Lmin=8, Lmax=10, verbose=1)
        df = m.get_samples_df(samples)
        df.to_pickle('samples_df_M{}.pickle'.format(M))
