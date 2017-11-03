import numpy as np
import gpflow
import VFF
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

from time import time

from config import *

dim = int(sys.argv[1])
rep = int(sys.argv[2])
num_basis = int(sys.argv[3])

num_inducing = num_basis ** dim
print('sparse: dimension {}, replicate {}, inducing {}'.format(dim, rep, num_inducing))

# data
data = np.load('data/data_dim{}_rep{}.npz'.format(dim, rep))
k = gpflow.kernels.Prod([gpflow.kernels.Matern32(1, active_dims=[i], lengthscales=lengthscale) for i in range(dim)])

# build sparse GP model
Z_grid = np.linspace(0, 1, num_basis) 
tmp = np.array(np.meshgrid(*[Z_grid for _ in range(dim)]))
Z = tmp.reshape(dim, num_basis ** dim).T

m = gpflow.sgpr.SGPR(data['Xtrain'], data['Ytrain'], Z=Z, kern=k)
m.likelihood.variance = noise_var

start = time()
marg_lik = m.compute_log_likelihood().squeeze()
# mean_log_pred = np.mean(m.predict_density(data['Xtest'], data['Ytest']))
mean_log_pred = 0*marg_lik
t = time() - start

file = open("results/sparse.csv","a") 
file.write("{},{},{},{},{},{}\n".format(dim, rep, num_inducing, marg_lik, mean_log_pred, t)) 
file.close() 

