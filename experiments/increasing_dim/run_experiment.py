import numpy as np
import subprocess

from config import *

import gpflow
import VFF


## initialise empty file
# file = open("results/full.csv","w") 
# file.write("dim,rep,marg_lik,mean_log_pred\n") 
# file.close() 

file = open("results/vff.csv","w") 
file.write("dim,rep,num_inducing,marg_lik,mean_log_pred,time\n") 
file.close() 

file = open("results/sparse.csv","w") 
file.write("dim,rep,num_inducing,marg_lik,mean_log_pred,time\n") 
file.close() 


## Generate Data
# subprocess.call(["python", "gen_data.py"])

## fit full GPR
# for dim in dimensions:
#     for rep in range(repeats):
#         subprocess.call(["python", "full.py", str(dim), str(rep)])

## fit VFF
for i, dim in enumerate(dimensions):
    for rep in range(repeats):
            for num_basis in num_Basis[i]:
                subprocess.call(["python", "vff.py", str(dim), str(rep), str(num_basis)])

## fit sparse
for i, dim in enumerate(dimensions):
    for rep in range(repeats):
            for num_basis in num_Basis[i]:
                subprocess.call(["python", "sparse.py", str(dim), str(rep), str(num_basis)])


