import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from run_pines import build_model, getLocations
from sklearn.neighbors import KernelDensity
# import matplotlib2tikz

X = getLocations()
Ms = [14, 16, 18, 20, 22, 24, 26, 28, 30]


def plot_model(m, sample_df, ax, gridResolution=64):
    intensities = []
    for _, s in sample_df.iterrows():
        m.set_parameter_dict(s)
        mu, _ = m.predict_y(m.X.value)
        intensities.append(mu)
    intensity = np.mean(intensities, 0)
    ax.imshow(np.flipud(intensity.reshape(gridResolution, gridResolution).T),
              interpolation='nearest', extent=[0, 1, 0, 1], cmap=plt.cm.viridis,
              vmin=0.005, vmax=0.18)

f, axes = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(12, 5))
for ax, M in zip(axes.flat, Ms):
    continue
    m = build_model(M)
    df = pd.read_pickle('samples_df_M{}.pickle'.format(M))
    # df = df.ix[::100]  # thin for speed
    plot_model(m, df, ax)
    ax.set_title(str(M))

# axes.flatten()[-1].plot(X[:, 0], X[:, 1], 'k.')
# matplotlib2tikz.save('pines_intensity.tikz')


# plot the convergence of the patermeters:
f, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12, 5))
keys = ['model.kerns.item0.lengthscales', 'model.kerns.item1.lengthscales']
titles = ['lengthscale (horz.)', 'lengthscale (vert)']
mins = [0, 0]
maxs = [0.4, 0.4]
for key, title, ax, xmin, xmax in zip(keys, titles, axes.flatten(), mins, maxs):
    for M in Ms:
        m = build_model(M)
        df = pd.read_pickle('samples_df_M{}.pickle'.format(M))
        ls = np.vstack(df[key])
        kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(ls)
        X_plot = np.linspace(xmin, xmax, 100)[:, None]
        ax.plot(X_plot, np.exp(kde.score_samples(X_plot)), label=str(M))
ax.legend()

# matplotlib2tikz.save('pines_lengthscale_convergence.tikz')
