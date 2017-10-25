import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
plt.ion()
# plt.close('all')


def plot_KL(d):
    d_full = pd.read_csv('results/full.csv', index_col=0)

    d = pd.merge(d, d_full, on=['rep', 'dim'])

    f, axes = plt.subplots(1, 5, figsize=(14, 5), sharex=True, sharey=True)
    for i, a in enumerate(axes):
        di = d[d.dim == (i+1)]
        a.set_title('dimensions={}'.format(i+1))
        for m in np.unique(di.num_inducing.values):
            dim = di[di.num_inducing == m]
            a.plot(dim.time, dim.marg_lik_y - dim.marg_lik_x, 'o', label='M={}'.format(int(m)))
            a.set_ylim(10**-2, 5e4)
            a.set_xlim(10**-1, 10**2)
            a.loglog()
    if i == 0:
        plt.legend()


def plot_pred(d):
    d_full = pd.read_csv('results/full.csv', index_col=0)

    d = pd.merge(d, d_full, on=['rep', 'dim'])

    f, axes = plt.subplots(1, 5, figsize=(14, 5), sharex=True, sharey=True)
    for i, a in enumerate(axes):
        di = d[d.dim == (i+1)]
        a.set_title('dimensions={}'.format(i+1))
        for m in np.unique(di.num_inducing.values):
            dim = di[di.num_inducing == m]
            a.plot(dim.time, -dim.mean_log_pred_x + dim.mean_log_pred_y, 'o', label='M={}'.format(int(m)))
            a.semilogx()
            #a.set_ylim(-0.02, 0.7)
            #a.set_xlim(10**-1, 10**5)
        plt.legend()

def plot_all(dim):
    d_full = pd.read_csv('results/full.csv', index_col=0)
    d_sparse_opt = pd.read_csv('results/sparse_opt.csv', index_col=0)
    d_sparse_kmeans = pd.read_csv('results/sparse_kmeans.csv', index_col=0)
    d_kron_opt = pd.read_csv('results/kron_opt.csv', index_col=0)
    d_kron = pd.read_csv('results/kron.csv', index_col=0)
    
    names = ['VFF', 'VFF opt', 'sparse', 'sparse opt']
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    for i, d in enumerate([d_kron, d_kron_opt, d_sparse_kmeans, d_sparse_opt]):
        df = pd.merge(d[d.dim==dim], d_full[d_full.dim==dim], on=['rep'])
        ax.plot(df.time,-df.mean_log_pred_x + df.mean_log_pred_y, 'o', label=names[i])
        ax.set_ylim((-0.02,1.6))
        ax.semilogx()
        ax.set_title('dimension {}'.format(dim))
    plt.legend()

for d in range(1,5):
    plot_all(d)

# plot_pred(pd.read_csv('results/sparse_opt.csv', index_col=0))
# plot_pred(pd.read_csv('results/sparse_kmeans.csv', index_col=0))
# plot_pred(pd.read_csv('results/kron_opt.csv', index_col=0))
# plot_pred(pd.read_csv('results/kron.csv', index_col=0))

# plot_KL(pd.read_csv('sparse_opt.csv', index_col=0))
# plot_KL(pd.read_csv('sparse_kmeans.csv', index_col=0))
# plot_KL(pd.read_csv('kron_opt.csv', index_col=0))
