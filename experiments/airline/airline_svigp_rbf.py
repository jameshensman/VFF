# Copyright 2016 James Hensman, Arno Solin
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


from __future__ import print_function
import numpy as np
import gpflow
import pandas as pd
import tensorflow as tf
import sys
import time

from scipy.stats import norm

# Import the data
data = pd.read_pickle('airline.pickle')

# Convert time of day from hhmm to minutes since midnight
data.ArrTime = 60*np.floor(data.ArrTime/100)+np.mod(data.ArrTime, 100)
data.DepTime = 60*np.floor(data.DepTime/100)+np.mod(data.DepTime, 100)


def subset(data, n):

    # Pick out the data
    Y = data['ArrDelay'].values
    names = ['Month', 'DayofMonth', 'DayOfWeek', 'plane_age', 'AirTime', 'Distance', 'ArrTime', 'DepTime']
    X = data[names].values

    # Shuffle the data and only consider a subset of it
    perm = np.random.permutation(len(X))
    X = X[perm]
    Y = Y[perm]
    XT = X[int(2*n/3):n]
    YT = Y[int(2*n/3):n]
    X = X[:int(2*n/3)]
    Y = Y[:int(2*n/3)]

    # Normalize Y scale and offset
    Ymean = Y.mean()
    Ystd = Y.std()
    Y = (Y - Ymean) / Ystd
    Y = Y.reshape(-1, 1)
    YT = (YT - Ymean) / Ystd
    YT = YT.reshape(-1, 1)

    # Normalize X on [0, 1]
    Xmin, Xmax = X.min(0), X.max(0)
    X = (X - Xmin) / (Xmax - Xmin)
    XT = (XT - Xmin) / (Xmax - Xmin)

    return X, Y, XT, YT

# Number of repetitions
repetitions = 10

# Sample sizes: [10000 100000 1000000 len(data)]
sample_size = [10000, 100000, 1000000, len(data)]

# MSE
mse = np.zeros([repetitions, len(sample_size)])
nlpd = np.zeros([repetitions, len(sample_size)])
tc = np.zeros([repetitions, len(sample_size)])
tt = np.zeros([repetitions, len(sample_size)])

# For repetitions
for i in range(repetitions):

    # Loop over the sample sizes
    for j in range(len(sample_size)):

        # Lock random seed
        np.random.seed(sample_size[j]+i)

        # Reset tensorflow
        tf.reset_default_graph()

        # Reset clocks
        tc0 = time.clock()
        tt0 = time.time()

        # Pick subset
        X, Y, XT, YT = subset(data, sample_size[j])

        # get inducing point by k-means. Use a smallish randomsubset for kmeans else it's very slow.
        from scipy.cluster import vq
        ind = np.random.permutation(X.shape[0])[:10000]
        Z, _ = vq.kmeans(X[ind, :], 500)

        # Set up the model
        m = gpflow.models.SVGP(
            kernel=gpflow.kernels.Matern32(lengthscales=np.ones(X.shape[1])),
            likelihood=gpflow.likelihoods.Gaussian(),
            inducing_variable=gpflow.inducing_variables.InducingPoints(Z),
            num_data=X.shape[0]
        )

        # Optimise the hyperparameters
        opt = tf.optimizers.Adam()
        maxiter = 100000
        data = tf.data.Dataset.from_tensor_slices((X, Y)).batch(int(1e3))
        obj = m.training_loss_closure(iter(data))
        for i in range(maxiter):
            if i % 11 == 0: print(i, obj())
            opt.minimize(obj, var_list=m.trainable_variables)

        # Evaluate test points in batches of 1e5
        mu, var = np.zeros([XT.shape[0], 1]), np.zeros([XT.shape[0], 1])
        for k in range(0, XT.shape[0], 100000):
            mu[k:k+100000], var[k:k+100000] = m.predict_y(XT[k:k+100000])

        # Calculate MSE
        mse[i, j] = ((mu-YT)**2).mean()
        print(X.shape[0], mse[i, j])

        # Calculate NLPD
        l = norm.logpdf(YT, loc=mu, scale=var ** 0.5)
        nlpd[i, j] = -np.mean(l)

        # Store time
        tc[i, j] = time.clock() - tc0
        tt[i, j] = time.time() - tt0

    # The results after this round
    print(mse[:i+1, :].mean(axis=0))
    print(mse[:i+1, :].std(axis=0))

print('MSE:')
print(mse.mean(axis=0))
print(mse.std(axis=0))

print('NLPD:')
print(nlpd.mean(axis=0))
print(nlpd.std(axis=0))

print('Timing (clock):')
print(tc.mean(axis=0))
print(tc.std(axis=0))

print('Timing (time):')
print(tt.mean(axis=0))
print(tt.std(axis=0))
