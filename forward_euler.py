from scipy import interpolate
import numpy as np
import utils


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
        x_i = x_i + V_i/steps #/steps # artificially large alpha to see the change: remove /steps
        # interpolate
        V_i = interpolate_n_d(x_0, V_0, x_i)

    # handle boundary conditions - I think it's ok to do this only once in the end
    dim = x_0.shape[1]
    img_shape = []
    for d in range(dim):
        img_shape.append(max(x_0[:, d]))
    x_i = utils.enforce_boundaries(x_i, img_shape)
    return x_i
