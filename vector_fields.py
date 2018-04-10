from __future__ import division
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances

# Produce control points (= grid) for a 2d grayscale image.
def get_points_2d(image, res):
    rows, columns = image.shape
    # Note: We index the points in a way that the columns change slowest, the rows fastest. This is
    # due to numpy array indexing (now consistent with the 'F' order for 3d.)
    grid_x, grid_y = np.mgrid[0:columns:res, 0:rows:res, ]
    grid = np.array((grid_x.flatten(), grid_y.flatten())).T
    return grid


# Produce control points (= grid) for a 3d grayscale image.
def get_points_3d(image, res):
    rows, columns, z = image.shape
    # Again, mgrid: slowest, middle, fastest changing index.
    grid_z, grid_x, grid_y = np.mgrid[0:z:res, 0:columns:res, 0:rows:res]
    grid = np.array((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
    return grid


# Wendland kernel
# Note that now this is a representation where we write k for k*J_dxd.
def kernel(x_i, x_j, c_sup):
    r = np.linalg.norm(x_i - x_j)/c_sup
    return max((1-r, 0))**4 * (4*r + 1)


def dist_kernel(r):
    return max((1-r, 0))**4 * (4*r + 1)


# Compute Gram matrix S (in a naive way).
def gram_matrix(function, grid):
    n, d = grid.shape
    S = np.zeros((d*n, d*n))
    for i in range(n):
        for j in range(i, n):
            for k in range(d):
                S[d*i+k][d*j+k] = function(grid[i], grid[j])
                S[d*j+k][d*i+k] = S[d*i+k][d*j+k]
    return sparse.csc_matrix(S)


# Quick computation of the Gram/evaluation matrices.
def evaluation_matrix(function, kernels, points, c_sup=200):
    vect_kernel = np.vectorize(dist_kernel)
    S = euclidean_distances(points, kernels)/c_sup
    m, n = S.shape
    unique = vect_kernel(S[:, 0])
    value_lookup = {dst: val for dst, val in zip(S[:,0], unique)}
    S[np.where(S == 1)] = 2
    inserted = []
    for d, val in value_lookup.items():
        if d < 1 and d not in inserted:
            S[np.where(S == d)] = val
            inserted.append(d)
    S[S>1] =0
    full_S = np.block([[np.eye(3)*S[i, j] for j in range(n)] for i in range(m)])
    return sparse.csc_matrix(full_S)


def make_random_V(S, d):
    nxd = S.shape[1]
    alpha = (np.random.rand(nxd) - 0.5)*2*10  # artificially large alpha to see the change
    # Change to Gaussian? In the real HMCS we will start with zeros
    lmda = S.dot(alpha)
    return lmda.reshape(-1, d)


def make_V(S, alpha, d):
    lmda = S.dot(alpha)
    return lmda.reshape(-1, d)
