from __future__ import print_function
import numpy as np
import GPflow
import pandas as pd
import time
import tensorflow as tf

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

# Number of repetitions: 10
repetitions = 10

# Sample sizes: [10000, 100000, 1000000, len(data)]
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

        # Do not do attempt to run the full model for large data sets
        if (j > 0):
            break

        # Set up the model
        m = GPflow.gpr.GPR(X, Y, kern=GPflow.kernels.RBF(X.shape[1], ARD=True))

        # Optimise the hyperparameters
        m.optimize(disp=1)

        # Evaluate test points in batches of 1e5
        mu, var = np.zeros([XT.shape[0], 1]), np.zeros([XT.shape[0], 1])
        for k in range(0, XT.shape[0], 100000):
            mu[k:k+100000], var[k:k+100000] = m.predict_y(XT[k:k+100000])

        # Calculate MSE
        mse[i, j] = ((mu-YT)**2).mean()

        # Calculate NLPD
        nlpd[i, j] = -np.mean(m.predict_density(XT, YT))

        # Store time
        tc[i, j] = time.clock() - tc0
        tt[i, j] = time.time() - tt0

    # The results after this round
    print(mse[:i+1].mean(axis=0))
    print(mse[:i+1].std(axis=0))

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
