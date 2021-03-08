import numpy as np

##########################
# config
dimensions = [1, 2, 3, 4]
repeats = 5

# num_inducing = [10, 50, 100, 200, 500]  # for sparse GP
# num_Basis = np.array([3, 5, 7])  # for VFF
num_Basis = [
    np.array([5, 9, 15, 21]),
    np.array([3, 7, 15, 21, 27, 35, 45]),
    np.array([3, 5, 7, 9, 11, 13]),
    np.array([3, 5, 7, 9]),
]

## num_Basis = [np.array([5, 9, 15, 21, 31]),
##              np.array([3, 5, 7, 9]),
##              np.array([3, 5, 7]),
##              np.array([3, 5])]

lengthscale = 0.2  # data are on [0, 1] ^ dim
noise_var = 0.1

num_train = 10000
num_test = 1000
