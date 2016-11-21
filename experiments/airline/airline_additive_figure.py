from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
import GPflow
import VFF
import pandas as pd

# Import the data
data = pd.read_pickle('airline.pickle')

# Convert time of day from hhmm to minutes since midnight
data.ArrTime = 60*np.floor(data.ArrTime/100)+np.mod(data.ArrTime, 100)
data.DepTime = 60*np.floor(data.DepTime/100)+np.mod(data.DepTime, 100)

# remove flights with silly negative delays (small negative delays are OK)
data = data[data.ArrDelay > -60]
# remove outlying flights in term of length
data = data[data.AirTime < 700]

# Pick out the data
Y = data['ArrDelay'].values
names = ['Month', 'DayofMonth', 'DayOfWeek', 'plane_age', 'AirTime', 'Distance', 'ArrTime', 'DepTime']
X = data[names].values

# normalize Y scale and offset
Ymean = Y.mean()
Ystd = Y.std()
Y = (Y - Ymean) / Ystd
Y = Y.reshape(-1, 1)

# normalize X on [0, 1]
Xmin, Xmax = X.min(0), X.max(0)
X = (X - Xmin) / (Xmax - Xmin)


def plot(m):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    Xtest = np.linspace(0, 1, 200)[:, None]
    mu, var = m.predict_components(Xtest)
    for i in range(mu.shape[1]):
        ax = axes.flatten()[i]
        Xplot = Xtest * (Xmax[i] - Xmin[i]) + Xmin[i]
        ax.plot(Xplot, mu[:, i], lw=2)
        ax.plot(Xplot, mu[:, i] + 2*np.sqrt(var[:, i]), 'b--', lw=1)
        ax.plot(Xplot, mu[:, i] - 2*np.sqrt(var[:, i]), 'b--', lw=1)
        ax.set_title(names[i])


if __name__ == '__main__':
    m = VFF.gpr.GPR_additive(X, Y, np.arange(30), np.zeros(X.shape[1]) - 2, np.ones(X.shape[1]) + 2,
                             [GPflow.kernels.Matern32(1) for i in range(X.shape[1])])
    m.optimize(disp=1)

    plot(m)

    plt.show()
