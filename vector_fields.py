from __future__ import division
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances

# Produce control points (= grid) for a 2d grayscale image.
def get_points_2d(image, res):
    rows, columns = image.shape
    # Note: We index the points in a way that the columns change slowest, the rows fastest. This is
    # due to numpy array indexing (now consistent with the 'F' order for 3d.)
    grid_x, grid_y = np.mgrid[0:columns:res, 0:rows:res]
    grid = np.array((grid_x.flatten(), grid_y.flatten())).T
    return grid


# Produce control points (= grid) for a 3d grayscale image.
def get_points_3d(image, res):
    rows, columns, z = image.shape
    # Again, mgrid: slowest, middle, fastest changing index.
    grid_z, grid_x, grid_y = np.mgrid[0:z:res, 0:columns:res, 0:rows:res]
    grid = np.array((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
    return grid


# Wendland kernel as a function of x and y
def kernel(x_i, x_j, c_sup):
    r = np.linalg.norm(x_i - x_j)/c_sup
    return max((1-r, 0))**4 * (4*r + 1)


# Wendland kernel as a function of r = norm(x-y)/c_sup
def dist_kernel(r):
    return max((1-r, 0))**4 * (4*r + 1)


# Quick computation of the Gram/evaluation matrices.
def evaluation_matrix(function, kernels, points, c_sup=200):
    dim = kernels.shape[1]
    vect_kernel = np.vectorize(dist_kernel)
    S = euclidean_distances(points, kernels)/c_sup
    m, n = S.shape
    unique = vect_kernel(S[:, 0])
    value_lookup = {dst: val for dst, val in zip(S[:,0], unique)}
    for d, val in value_lookup.items():
        if d <= 1:
            # use minus sign to 'mark' already done entries
            S[np.where(S == d)] = - val
    S[S > 1] = 0
    S = -S
    print('Computations done.')
    full_S = np.block([[np.eye(dim)*S[i, j] for j in range(n)] for i in range(m)])
    # TODO: we can avoid this blowing up by reshaping S and alpha as discussed today (25.04.)
    print('Blowing up done.')
    return sparse.csc_matrix(full_S)


def make_V(S, alpha, d):
    alpha = alpha.flatten()
    lmda = S.dot(alpha)
    return lmda.reshape(-1, d)
