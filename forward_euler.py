from scipy import sparse
import time

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
    start = time.time()
    if (compute_gradient):
        S_i = vector_fields.evaluation_matrix_blowup(kernels, x_0, c_sup, dim)
    else:
        S_i = vector_fields.evaluation_matrix(kernels, x_0, c_sup, dim)
    #print("FE -- initial S_i ", (time.time() - start) / 60)
    start = time.time()
    V_i = vector_fields.make_V(S_i, alpha, dim)
    #print("FE -- initial V_i ", (time.time() - start) / 60)
    x_i = x_0
    dphi_dalpha_i = sparse.csc_matrix((S_i.shape[0], S_i.shape[1]))
    
    for i in range(steps):
        if compute_gradient:
            start = time.time()
            dv_dphit_i = gradient.dv_dphit(x_i, kernels, alpha, c_sup, dim)
            print("FE -- dv_dphi_i ", (time.time() - start) / 60)
            start = time.time()
            dphi_dalpha_i = gradient.next_dphi_dalpha(S_i, dv_dphit_i, dphi_dalpha_i, steps, dim)
            print("FE -- dphi_dalpha_i ", (time.time() - start) / 60)
        start = time.time()   
        # Make a step
        x_i = x_i + V_i / steps

        # Compute evaluatation matrix based on updated evaluation points
        if (i < steps - 1):
            if (compute_gradient):
                S_i = vector_fields.evaluation_matrix_blowup(kernels, x_i, c_sup, dim)
            else:
                S_i = vector_fields.evaluation_matrix(kernels, x_i, c_sup, dim)
            V_i = vector_fields.make_V(S_i, alpha, dim)

        print("FE -- Euler step ", (time.time() - start) / 60)
    # Enforce boundary conditions
    img_shape = []
    for d in range(dim):
        img_shape.append(max(x_0[:, d]))
    x_1 = enforce_boundaries(x_i, img_shape, dim)

    if compute_gradient:
        return x_1, dphi_dalpha_i
    else:
        return x_1
