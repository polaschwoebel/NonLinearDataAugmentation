from __future__ import division
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances
import time

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
        S_full = sparse.lil_matrix((3 * m, 3 * n), dtype = np.float32)
        #S_full = np.zeros((3 * m, 3 * n))
        S_full[0::3, 0::3] = S
        S_full[1::3, 1::3] = S
        S_full[2::3, 2::3] = S
    else:
        S_full = np.zeros((2 * m, 2 * n))
        S_full[0::2, 0::2] = S
        S_full[1::2, 1::2] = S
    return S_full.tocsc()

# Create evaluation matrix given kernel centers (grid points), evaluation points
# and kernel support
def evaluation_matrix(kernels, points, c_sup, dim):
    dim = kernels.shape[1]
    vect_kernel = np.vectorize(dist_kernel)
    start = time.time()
    S = euclidean_distances(points, kernels) / c_sup
    #print("VEC -- euc dist ", (time.time() - start) / 60)
    # Mark entries with 0 kernel support
    start = time.time()
    S[np.where(S > 1)] = -1
    non_zero_indices = np.where(S >= 0)
    #print("VEC -- S[np.where(S > 1)] and np.where(S>=0) ", (time.time() - start) / 60)
    # Evaluate kernel at points within support
    start = time.time()
    S[non_zero_indices] = vect_kernel(S[non_zero_indices])
    #print("VEC -- S[non_zero] = vect_kernel ", (time.time() - start) / 60)
    start = time.time()
    S[np.where(S == -1)] = 0
    #print("VEC -- S[np.where(S == -1)] = 0 ", (time.time() - start) / 60)
    start = time.time()
    #full_S = blowup_S_old(S, dim)
    #print("VEC -- blowup ", (time.time() - start) / 60)
    return sparse.csc_matrix(S)

def evaluation_matrix_blowup(kernels, points, c_sup, dim):
    dim = kernels.shape[1]
    vect_kernel = np.vectorize(dist_kernel)
    start = time.time()
    S = euclidean_distances(points, kernels) / c_sup
    #print("VEC -- euc dist ", (time.time() - start) / 60)
    # Mark entries with 0 kernel support
    start = time.time()
    S[np.where(S > 1)] = -1
    non_zero_indices = np.where(S >= 0)
    #print("VEC -- S[np.where(S > 1)] and np.where(S>=0) ", (time.time() - start) / 60)
    # Evaluate kernel at points within support
    start = time.time()
    S[non_zero_indices] = vect_kernel(S[non_zero_indices])
    #print("VEC -- S[non_zero] = vect_kernel ", (time.time() - start) / 60)
    start = time.time()
    S[np.where(S == -1)] = 0
    #print("VEC -- S[np.where(S == -1)] = 0 ", (time.time() - start) / 60)
    start = time.time()
    full_S = blowup_S(S, dim)
    #print("VEC -- blowup ", (time.time() - start) / 60)
    return full_S


# Create velocity field by weighing kernels by alphas
def make_V(S, alpha, d):
    alpha = alpha.flatten()
    if (S.shape[1] == alpha.shape[0]):
        lmda = S.dot(alpha)
        return lmda.reshape(-1, d)
    else:
        alpha = alpha.reshape(-1, d)
        return S.dot(alpha)


