import numpy as np
import gpflow
import VFF
import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from time import time

from config import *

dim = int(sys.argv[1])
rep = int(sys.argv[2])
num_basis = int(sys.argv[3])
a = float(sys.argv[4])

num_inducing = num_basis ** dim
print("vff: dimension {}, replicate {}, inducing {}, a {}".format(dim, rep, num_inducing, a))

# data
data = np.load("data/data_dim{}_rep{}.npz".format(dim, rep))
k_list = [gpflow.kernels.Matern32(1, lengthscales=lengthscale) for i in range(dim)]

b = (1 - a) * np.ones(dim)
a = a * np.ones(dim)

m = VFF.gpr.GPRKron(
    data["Xtrain"], data["Ytrain"], np.arange((num_basis + 1) / 2), a, b, kerns=k_list
)
m.likelihood.variance = noise_var
m.likelihood.fixed = True
m.fixed = True

start = time()
marg_lik = m.compute_log_likelihood().squeeze()
mean_log_pred = np.mean(m.predict_density(data["Xtest"], data["Ytest"]))
t = time() - start

file = open("results/vff.csv", "a")
file.write(
    "{},{},{},{},{},{},{}\n".format(dim, rep, num_inducing, marg_lik, mean_log_pred, t, a[0])
)
file.close()
