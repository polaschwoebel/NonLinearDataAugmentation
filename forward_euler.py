from scipy import interpolate
import numpy as np


# wrapper for scipy interpolation so that it can handle the d-dimensional vectorfields (d=2 or d=3)
def interpolate_n_d(x_0, V_0, x_i):
    n, d = x_i.shape
    V_i = np.empty((n, d))
    for dim in range(d):
        V_i[:, dim] = interpolate.griddata(x_0, V_0[:, dim], x_i, fill_value=0)
    return V_i


def integrate(x_0, V_0, steps):
    V_i = V_0
    x_i = x_0
    for _ in range(steps):
        # make a step
        x_i = x_i + V_i #/steps # artificially large alpha to see the change: remove /steps
        # interpolate
        V_i = interpolate_n_d(x_0, V_0, x_i)
    return x_i
