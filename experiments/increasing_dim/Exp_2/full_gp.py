import GPflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config import dimensions, noise_var, prodkern, repeats

results = pd.DataFrame()

for dim in dimensions:
    plt.figure()
    for r in range(repeats):
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

