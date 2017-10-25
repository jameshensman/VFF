import GPflow
dimensions = [1, 2, 3, 4, 5]
repeats = 5
num_inducing = [10, 50, 100, 200, 500]  # for sparse GP
num_freqs = [5, 10, 15, 20, 50]  # for VFF

lengthscale = 0.3  # data are on [0, 1] ^ D
noise_var = 0.1

num_train = 3000
num_test = 1000


def prodkern(dim):
    return GPflow.kernels.Prod([GPflow.kernels.Matern32(1, active_dims=[i], lengthscales=lengthscale)
                                for i in range(dim)])
