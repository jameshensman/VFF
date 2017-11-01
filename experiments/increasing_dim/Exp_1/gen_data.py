import numpy as np
import gpflow

from config import *

def prodkern(dim):
    return gpflow.kernels.Prod([gpflow.kernels.Matern32(1, active_dims=[i], lengthscales=lengthscale)
                                for i in range(dim)])

##########################
# gen_data

np.random.seed(0)
for dim in dimensions:
    k = prodkern(dim) + gpflow.kernels.White(1, variance=noise_var)
    for r in range(repeats):
        print('gen_data: dimension{} repeat{}'.format(dim,r))
        X = np.random.rand(num_train + num_test, dim)
        K = k.compute_K_symm(X)
        L = np.linalg.cholesky(K)
        Y = np.dot(L, np.random.randn(num_train + num_test, 1))
        Ytrain, Ytest = Y[:num_train], Y[num_train:]
        Xtrain, Xtest = X[:num_train], X[num_train:]

        np.savez('data/data_dim{}_rep{}.npz'.format(dim, r),
                 Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest)

