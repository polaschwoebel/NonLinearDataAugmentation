import numpy as np


# Produce control points (= grid) for a 2d grayscale image.
def get_points_2d(image, res):
    x, y = image.shape
    grid_y, grid_x = np.mgrid[0:y:res, 0:x:res]
    grid = np.array((grid_y.flatten(), grid_x.flatten())).T
    return grid


# Produce control points (= grid) for a 3d grayscale image.
def get_points_3d(image, res):
    y, x, z = image.shape
    grid_y, grid_x, grid_z = np.mgrid[0:x:100, 0:y:100, 0:z:100]
    grid = np.array((grid_y.flatten(), grid_x.flatten(), grid_z.flatten())).T
    return grid


# Wendland kernel
# Note that now this is a representation where we write k for k*J_dxd.
# TODO: Find proper scaling constant c_sup depending on the support and thus grid.
def kernel(x_i, x_j, c_sup):
    r = np.linalg.norm(x_i - x_j)/c_sup
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
    return S


def evaluation_matrix(function, kernel_grid, points):
    n, d = kernel_grid.shape
    m = points.shape[0]
    S = np.zeros((d*m, d*n))
    for i in range(m):
        for j in range(n):
            for k in range(d):
                S[d*i+k][d*j+k] = function(points[i], kernel_grid[j])
    return S


def make_random_V(S, d):
    nxd = S.shape[1]
    alpha = (np.random.rand(nxd) - 0.5)*2*10 # artificially large alpha to see the change
    # change to Gaussian?
    lmda = S.dot(alpha)
    return lmda.reshape(-1, d)
