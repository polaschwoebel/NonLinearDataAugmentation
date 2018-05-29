import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse
import matlab.engine
import matlab

# Compute the spacial Jacobian
def dv_dphit(phi_t, kernels, alpha, c_sup, dim):
    alpha = alpha.reshape((-1, dim)).T

    distances = euclidean_distances(phi_t, kernels) / c_sup
    m, n = distances.shape

    # Find evaluation- and kernel point pairs with non-zero kernel evaluations
    dist_smaller_1 = np.where(distances < 1)

    # Compute the kernel derivative w.r.t. evaluation points
    Kdev = np.zeros((kernels.shape[0], dim * phi_t.shape[0]))
    for i in range(len(dist_smaller_1[0])):
        
        # Retrieve indices for non-zero kernel derivative
        m_c = dist_smaller_1[0][i]
        n_c = dist_smaller_1[1][i]
        diff = phi_t[m_c, :] - kernels[n_c, :]
        
        # Analytically derived Wendland kernel derivative
        Kdev[n_c ,dim * m_c : dim * m_c + dim] = diff * (1 - np.linalg.norm(diff) 
        / c_sup) ** 3 * (-20 / (c_sup ** 2))

    # Compute velocity derivative by multiplying alpha
    Vdev = sparse.csc_matrix.dot(sparse.csc_matrix(alpha), sparse.csc_matrix(Kdev))
    Vdev = Vdev.tolil()
    Vdev_full = sparse.lil_matrix((dim * m, dim * m))
    for i in range(m):
        Vdev_full[i * dim : (i+1) * dim, i * dim : (i+1) * dim] = Vdev[:, 
                  i * dim : (i+1) * dim]
    return Vdev_full

# Compute dphi_dalpha for next Forward Euler step by the recursive definition
def next_dphi_dalpha(S, dv_dphit, prev_dphi_dalpha, step_size):
    m, n = S.shape
    identity = sparse.identity(m)
    dphi_dalpha = (identity + 1/step_size * dv_dphit).dot(prev_dphi_dalpha) + 1/step_size * S
    return dphi_dalpha

# Computation of the image gradient. Returns the full Jacobian matrix of 
# dimension 3m x 3m where m is number of evaluation points
def dIm_dphi(img, eng, spline_rep, phi, res):
    phi_x = matlab.double(phi[:,0].tolist())
    phi_y = matlab.double(phi[:,1].tolist())
    
    # Use spline representation of image to extract derivatives at phi
    imres = img.shape[0]
    dev1 = np.array(eng.eval_dev1(spline_rep, phi_x, phi_y, imres), dtype=np.float32)
    dev2 = np.array(eng.eval_dev2(spline_rep, phi_x, phi_y, imres), dtype=np.float32)
    dev1[np.isnan(dev1)] = 0
    dev2[np.isnan(dev2)] = 0
    gradients_all_dims = [dev2, dev1]
    gradient_array = np.dstack([dim_arr.flatten(order='F') for dim_arr in gradients_all_dims])[::-1][0]
    block_diag = sparse.block_diag(gradient_array)
    return block_diag

def dED_dphit(im_source, im_target, trans_points, points, dIm1_dphi1, dim):
    if dim == 3:
        target_points = im_target[points[:, 1], points[:, 0], points[:, 2]]
    else:
        target_points = im_target[points[:, 1], points[:, 0]]
    diff = sparse.csr_matrix(trans_points - target_points)
    return sparse.csc_matrix(2 * diff.dot(dIm1_dphi1))

def dER_dalpha(G, alpha):
    alpha = alpha.reshape((-1, 1))
    return sparse.csc_matrix(2 * G.dot(alpha))

