from scipy import interpolate
import numpy as np
import utils
import gradient
import vector_fields

# WILL BE REPLACED ONCE CHANGED BELOW
# wrapper for scipy interpolation so that it can handle the d-dimensional vectorfields (d=2 or d=3)
def interpolate_n_d(x_0, V_0, x_i):
    n, d = x_i.shape
    V_i = np.empty((n, d))
    for dim in range(d):
        V_i[:, dim] = interpolate.griddata(x_0, V_0[:, dim], x_i, fill_value=0)
    return V_i


def integrate(x_0, kernels, alpha, S, c_sup, steps=10, compute_gradient=True):
    dim = x_0.shape[1]
    V_i = vector_fields.make_V(S, alpha.reshape((alpha.size, -1)), dim)
    x_i = x_0
    du_dalpha_i = S
    for _ in range(steps):
        print('Computing step', _)
        # make a step
        x_i = x_i + V_i/steps
        # Note: first S_i computation could be avoided since S_i is passed
        S_i = vector_fields.evaluation_matrix(lambda x1, x2:
                                              vector_fields.kernel(x1, x2, c_sup), kernels, x_i, c_sup, dim)
        V_i = vector_fields.make_V(S_i, alpha.reshape((alpha.size, -1)), dim)
        print('Computing done. Now gradient, if desired.')
        if compute_gradient:
            # gradient computations
            dv_dphit_i = gradient.dv_dphit(x_i, kernels, alpha, c_sup=c_sup)
            du_dalpha_i = gradient.next_dphi_dalpha(S_i, dv_dphit_i, du_dalpha_i, steps)
            print('Gradient.')

    # boundary conditions
    img_shape = []
    for d in range(dim):
        img_shape.append(max(x_0[:, d]))
    x_1 = utils.enforce_boundaries(x_i, img_shape)

    if compute_gradient:
        du_dalpha_1 = du_dalpha_i
        return x_1, du_dalpha_1
    else:
        return x_1
