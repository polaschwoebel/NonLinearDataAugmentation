from scipy import sparse

import gradient
import vector_fields

def enforce_boundaries(phi, img_shape, dim):
    # make sure we are inside the image
    phi[:, 1] = phi[:, 1].clip(0, img_shape[1])
    phi[:, 0] = phi[:, 0].clip(0, img_shape[0])
    # 3d case
    if dim == 3:
        phi[:, 2] = phi[:, 2].clip(0, img_shape[2])
    return phi

def integrate(x_0, kernels, alpha, c_sup, dim, steps = 10, compute_gradient = True):
    S_i = vector_fields.evaluation_matrix(kernels, x_0, c_sup, dim)
    V_i = vector_fields.make_V(S_i, alpha, dim)
    x_i = x_0
    dphi_dalpha_i = sparse.csc_matrix(S_i.shape)

    for i in range(steps):
        if compute_gradient:
            print('FE -- compute dv_dphit_i')
            dv_dphit_i = gradient.dv_dphit_old_parallel(x_i, kernels, alpha, c_sup, dim)
            print('FE -- compute dphit_dalpha_i')
            dphi_dalpha_i = gradient.next_dphi_dalpha(S_i, dv_dphit_i, dphi_dalpha_i, steps)

        # Make a step
        x_i = x_i + V_i / steps

        # Compute evaluatation matrix based on updated evaluation points
        S_i = vector_fields.evaluation_matrix(kernels, x_i, c_sup, dim)
        V_i = vector_fields.make_V(S_i, alpha, dim)

    # Enforce boundary conditions
    img_shape = []
    for d in range(dim):
        img_shape.append(max(x_0[:, d]))
    x_1 = enforce_boundaries(x_i, img_shape, dim)

    if compute_gradient:
        return x_1, dphi_dalpha_i
    else:
        return x_1
