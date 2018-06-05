from __future__ import division
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances


# Produce grid points for a 2d grayscale image
def get_points_2d(image, res):
    rows, columns = image.shape
    grid_x, grid_y = np.mgrid[0:columns:res, 0:rows:res]
    grid = np.array((grid_x.flatten(), grid_y.flatten())).T
    return grid

# Produce grid points for a 3d grayscale image
def get_points_3d(image, res):
    rows, columns, z = image.shape
    grid_z, grid_x, grid_y = np.mgrid[0:z:res, 0:columns:res, 0:rows:res]
    grid = np.array((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
    return grid

# Wendland kernel as a function of r = norm(x-y)/c_sup
def dist_kernel(r):
    return max((1-r, 0))**4 * (4*r + 1)

def blowup_S(S, dim):
    (m, n) = S.shape
    if dim == 3:
        #S_full = np.zeros((3 * m, 3 * n))
        S_full = sparse.lil_matrix((3 * m, 3 * n), dtype = np.float32)
        S_full[0::3, 0::3] = S
        S_full[1::3, 1::3] = S
        S_full[2::3, 2::3] = S
    else:
        S_full = sparse.lil_matrix((2 * m, 2 * n), dtype = np.float32)
        S_full[0::2, 0::2] = S
        S_full[1::2, 1::2] = S
    return S_full.tocsc()

# Create evaluation matrix given kernel centers (grid points), evaluation points
# and kernel support
def evaluation_matrix(kernels, points, c_sup, dim):
    dim = kernels.shape[1]
    vect_kernel = np.vectorize(dist_kernel)
    S = euclidean_distances(points, kernels) / c_sup
    
    # Mark entries with 0 kernel support
    S[np.where(S > 1)] = -1
    non_zero_indices = np.where(S >= 0)
    
    # Evaluate kernel at points within support
    S[non_zero_indices] = vect_kernel(S[non_zero_indices])
    S[np.where(S == -1)] = 0
    full_S = blowup_S(S, dim)
    #return sparse.csc_matrix(full_S)
    return full_S

# Create velocity field by weighing kernels by alphas
def make_V(S, alpha, d):
    alpha = alpha.flatten()
    lmda = S.dot(alpha)
    return lmda.reshape(-1, d)
