import GPflow
import numpy as np
from time import time
from sklearn.cluster import KMeans
import pandas as pd

from config import dimensions, noise_var, prodkern, repeats, num_inducing


results = pd.DataFrame()

for dim in dimensions:
    for r in range(repeats):
        data = np.load('data_dim{}_rep{}.npz'.format(dim, r))
        for M in num_inducing:
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
            results.to_csv('sparse_opt.csv')

