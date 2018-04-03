from __future__ import division
import numpy as np
from scipy import sparse

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


def evaluation_matrix(function, kernel_grid, points):
    n, d = kernel_grid.shape
    m = points.shape[0]
    S = np.zeros((d*m, d*n))
    for i in range(m):
        for j in range(n):
            for k in range(d):
                S[d*i+k][d*j+k] = function(points[i], kernel_grid[j])
    return sparse.csc_matrix(S)


def make_random_V(S, d):
    nxd = S.shape[1]
    alpha = (np.random.rand(nxd) - 0.5)*2*10  # artificially large alpha to see the change
   # alpha_symmetric = np.concatenate((alpha, alpha[::-1]))
    # Change to Gaussian? In the real HMCS we will start with zeros
    lmda = S.dot(alpha)
    return lmda.reshape(-1, d)


def make_V(S, alpha, d):
    lmda = S.dot(alpha)
    return lmda.reshape(-1, d)


####################### WIP - might not be used in the end
def evaluation_matrix_efficient(function, kernel_grid, points, c_sup, kernel_res=100, points_res=2): # kernel_res
    #kernel_res = kernel_grid[1][0]
    unique = np.zeros(c_sup//points_res)
    for i in range(c_sup//points_res):
        unique[i] = function(points[i], np.array([0, 0]))
    n, d = kernel_grid.shape
    m = points.shape[0]
    S = np.zeros((m, n))
    for i in range(n):
        #print(i)
        # lower diagonal
        S[(kernel_res//points_res)*i: min(m, (kernel_res//points_res)*i + (c_sup//points_res)), i] = unique
        #S[(c_sup//points_res) * i: min((c_sup//points_res) * (i + 1), m), i] = unique
        # upper diagonal
        # reverse the order and take everything up to the one
        unique_upper = unique[::-1][:-1]
        #print(kernel_res, points_res)
        #print('i is now:', i, 'len(unique_upper):', len(unique_upper), 'resolution ratio:', (kernel_res//points_res)*i)
        cut_start = abs(min(0, (kernel_res//points_res)*i - len(unique_upper)))
        unique_upper_cut = unique_upper[cut_start:]
        if i == 1:
            print(len(unique), cut_start)
            return
        S[max((kernel_res//points_res)*i - len(unique_upper), 0): (kernel_res//points_res)*i, i] = unique_upper_cut
        #S[(kernel_res//points_res)*i - len(unique_upper): min(m, (kernel_res//points_res)*i + (c_sup//points_res)), i]
        #S[(c_sup//points_res) * i: min((c_sup//points_res) * (i + 1), m), min(i+1, n-1)] = unique[::-1]
    # expand to make block matrices # TODO: this is s bit slow :(
    full_S = np.zeros((d*m, d*n))
    for i in range(m):
        for j in range(n):
            #print(d*j, d*j+d)
            #print(full_S[d*i:(d*i+d)][d*j:(d*j+d)].shape)
            full_S[d*i:(d*i+d), d*j:(d*j+d)] = S[i][j]*np.identity(d)
    return full_S
    # TODO: OK in the rows, but not in the columns :(
