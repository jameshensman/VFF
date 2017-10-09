import numpy as np
import pandas as pd
from time import time
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import GPflow
import VFF

##########################
# config
dimensions = [1, 2, 3, 4, 5]
repeats = 5
#num_inducing = [10, 50, 100, 200, 500]  # for sparse GP
num_freqs = np.array([1, 2, 3, 4, 5])  # for VFF

lengthscale = 0.3  # data are on [0, 1] ^ D
noise_var = 0.1

num_train = 5000
num_test = 1000

def prodkern(dim):
    return GPflow.kernels.Prod([GPflow.kernels.Matern32(1, active_dims=[i], lengthscales=lengthscale)
                                for i in range(dim)])

##########################
# gen_data
np.random.seed(0)
for dim in dimensions:
    for r in range(repeats):
        # draw a data set
        X = np.random.rand(num_train + num_test, dim)
        k = prodkern(dim) + GPflow.kernels.White(1, variance=noise_var)
        K = k.compute_K_symm(X)
        L = np.linalg.cholesky(K)
        Y = np.dot(L, np.random.randn(num_train + num_test, 1))
        Ytrain, Ytest = Y[:num_train], Y[num_train:]
        Xtrain, Xtest = X[:num_train], X[num_train:]

        np.savez('data/data_dim{}_rep{}.npz'.format(dim, r),
                 Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest)


#########################
## full_gp
results = pd.DataFrame()

for dim in dimensions:
    plt.figure()
    for r in range(repeats):
        print('full GP replicate ',r,'/',repeats)
        data = np.load('data/data_dim{}_rep{}.npz'.format(dim, r))
        # build full GP model
        m = GPflow.gpr.GPR(data['Xtrain'], data['Ytrain'], kern=prodkern(dim))
        m.likelihood.variance = noise_var
        marg_lik = m.compute_log_likelihood().squeeze()
        mean_log_pred = np.mean(m.predict_density(data['Xtest'], data['Ytest']))

        results = results.append(dict(dim=dim, rep=r, marg_lik=marg_lik,
                                      mean_log_pred=mean_log_pred),
                                 ignore_index=True)

        # do this inside the loop so we can get partial results if something crashes
        results.to_csv('results/full.csv')

        plt.plot(m.predict_f(data['Xtest'])[0], data['Ytest'], 'x')

##########################
# kron
results = pd.DataFrame()

for dim in dimensions:
    for r in range(repeats):
        print('kron replicate ',r,'/',repeats)
        data = np.load('data/data_dim{}_rep{}.npz'.format(dim, r))
        for M in num_freqs:
            if (M ** dim) > 1e5:
                continue
            # build kronecker GP model
            a, b = -0.5 * np.ones(dim), 1.5 * np.ones(dim)
            m = VFF.vgp.VGP_kron(data['Xtrain'], data['Ytrain'], np.arange(M), a, b,
                                 kerns=prodkern(dim).kern_list,
                                 likelihood=GPflow.likelihoods.Gaussian(),
                                 use_two_krons=True)
            m.likelihood.variance = noise_var

            # only optimize q(u)
            m.kerns.fixed = True
            m.likelihood.fixed = True

            start = time()
            m.optimize()
            marg_lik = m.compute_log_likelihood().squeeze()
            mean_log_pred = np.mean(m.predict_density(data['Xtest'], data['Ytest']))
            t = time() - start

            results = results.append(dict(dim=dim, rep=r, marg_lik=marg_lik,
                                          mean_log_pred=mean_log_pred, time=t,
                                          num_inducing=M),
                                     ignore_index=True)

            # do this inside the loop so we can get partial results if something crashes
            results.to_csv('results/kron.csv')

##########################
# kron_opt
results = pd.DataFrame()

for dim in dimensions:
    for r in range(repeats):
        print('kron_opt replicate ',r,'/',repeats)
        data = np.load('data/data_dim{}_rep{}.npz'.format(dim, r))
        for M in num_freqs:
            if (M ** dim) > 1e5:
                continue
            # build kronecker GP model
            a, b = -0.5 * np.ones(dim), 1.5 * np.ones(dim)
            m = VFF.vgp.VGP_kron(data['Xtrain'], data['Ytrain'], np.arange(M), a, b,
                                 kerns=prodkern(dim).kern_list,
                                 likelihood=GPflow.likelihoods.Gaussian(),
                                 use_two_krons=True)
            m.likelihood.variance = noise_var

            start = time()
            m.optimize()
            marg_lik = m.compute_log_likelihood().squeeze()
            mean_log_pred = np.mean(m.predict_density(data['Xtest'], data['Ytest']))
            t = time() - start

            results = results.append(dict(dim=dim, rep=r, marg_lik=marg_lik,
                                          mean_log_pred=mean_log_pred, time=t,
                                          num_inducing=M),
                                     ignore_index=True)

            results.to_csv('results/kron_opt.csv')



##########################
# Sparse
results = pd.DataFrame()

for dim in dimensions:
    for r in range(repeats):
        print('Sparse replicate ',r,'/',repeats)
        data = np.load('data/data_dim{}_rep{}.npz'.format(dim, r))
        num_inducing = (2*num_freqs-1)**dim
        for M in num_inducing:
            if M < num_train: 
                # build sparse GP model
                Z = KMeans(n_clusters=M).fit(data['Xtrain']).cluster_centers_
                m = GPflow.sgpr.SGPR(data['Xtrain'], data['Ytrain'], Z=Z, kern=prodkern(dim))
                m.likelihood.variance = noise_var

                start = time()
                marg_lik = m.compute_log_likelihood().squeeze()
                mean_log_pred = np.mean(m.predict_density(data['Xtest'], data['Ytest']))
                t = time() - start

                results = results.append(dict(dim=dim, rep=r, marg_lik=marg_lik,
                                              mean_log_pred=mean_log_pred, time=t,
                                              num_inducing=M),
                                         ignore_index=True)

                # do this inside the loop so we can get partial results if something crashes
                results.to_csv('results/sparse_kmeans.csv')



##########################
# Sparse GP opt 
results = pd.DataFrame()

for dim in dimensions:
    for r in range(repeats):
        print('sparse opt replicate ',r,'/',repeats)
        data = np.load('data/data_dim{}_rep{}.npz'.format(dim, r))
        num_inducing = (2*num_freqs-1)**dim
        for M in num_inducing:
            if M < num_train: 
                # build sparse GP model
                Z = KMeans(n_clusters=M).fit(data['Xtrain']).cluster_centers_
                m = GPflow.sgpr.SGPR(data['Xtrain'], data['Ytrain'], Z=Z, kern=prodkern(dim))
                m.likelihood.variance = noise_var

                # only optimize Z
                m.kern.fixed = True
                m.likelihood.fixed = True

                start = time()
                m.optimize()
                marg_lik = m.compute_log_likelihood().squeeze()
                mean_log_pred = np.mean(m.predict_density(data['Xtest'], data['Ytest']))
                t = time() - start

                results = results.append(dict(dim=dim, rep=r, marg_lik=marg_lik,
                                              mean_log_pred=mean_log_pred, time=t,
                                              num_inducing=M),
                                         ignore_index=True)

                # do this inside the loop so we can get partial results if something crashes
                results.to_csv('results/sparse_opt.csv')


##########################
# 
