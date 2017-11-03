import numpy as np
import sys
import gpflow
import VFF
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

from time import time

from config import *

dim = int(sys.argv[1])
rep = int(sys.argv[2])

print('full GP: dimension {}, replicate {}'.format(dim, rep))

# data
data = np.load('data/data_dim{}_rep{}.npz'.format(dim, rep))

# full_gp
def prodkern(dim):
    return gpflow.kernels.Prod([gpflow.kernels.Matern32(1, active_dims=[i], lengthscales=lengthscale)
                                for i in range(dim)])
k = prodkern(dim)
m = gpflow.gpr.GPR(data['Xtrain'], data['Ytrain'], kern=k)
m.likelihood.variance = noise_var
data = np.load('data/data_dim{}_rep{}.npz'.format(dim, rep))
marg_lik = m.compute_log_likelihood().squeeze()
mean_log_pred = np.mean(m.predict_density(data['Xtest'], data['Ytest']))

file = open("results/full.csv","a") 
file.write("{},{},{},{}\n".format(dim, rep, marg_lik, mean_log_pred)) 
file.close() 
