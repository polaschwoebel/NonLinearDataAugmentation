from __future__ import division
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
    alpha = (np.random.rand(nxd) - 0.5)*2*10  # artificially large alpha to see the change
    # change to Gaussian?
    lmda = S.dot(alpha)
    return lmda.reshape(-1, d)


def enforce_boundaries(V, img_shape):
    # make sure we are inside the image
    V[:, 1] = V[:, 1].clip(0, img_shape[0])
    V[:, 0] = V[:, 0].clip(0, img_shape[1])
    return V


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
