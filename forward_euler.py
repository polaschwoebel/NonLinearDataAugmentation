from scipy import interpolate
import numpy as np
import utils
import gradient
import vector_fields

# wrapper for scipy interpolation so that it can handle the d-dimensional vectorfields (d=2 or d=3)
def interpolate_n_d(x_0, V_0, x_i):
    n, d = x_i.shape
    V_i = np.empty((n, d))
    for dim in range(d):
        V_i[:, dim] = interpolate.griddata(x_0, V_0[:, dim], x_i, fill_value=0)
    return V_i


def integrate(x_0, kernels, alpha, S, steps=10):
    V_i = vector_fields.make_V(S, alpha.reshape((alpha.size, -1)), 3)
    x_i = x_0
    du_dalpha_i = S
    for _ in range(steps):
        print('Computing step', _)
        # make a step
        x_i = x_i + V_i/steps
        # interpolate
        V_i = interpolate_n_d(x_0, V_i, x_i)

        # gradient computations
        dv_dphit_i = gradient.dv_dphit(x_i, kernels, alpha, c_sup=200)
        du_dalpha_i = gradient.next_du_dalpha(S, dv_dphit_i, du_dalpha_i, steps)

    # handle boundary conditions - I think it's ok to do this only once in the end
    # still now with gradient?
    dim = x_0.shape[1]
    img_shape = []
    for d in range(dim):
        img_shape.append(max(x_0[:, d]))
    x_i = utils.enforce_boundaries(x_i, img_shape)

    return x_i, du_dalpha_i
