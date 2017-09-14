import GPflow
import numpy as np

from config import dimensions, num_train, num_test, noise_var, prodkern, repeats

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

        np.savez('data_dim{}_rep{}.npz'.format(dim, r),
                 Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest)


