import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cv2
import numpy as np
import vector_fields
import forward_euler


def plot_grid_2d(grid, filename):
    plt.clf()
    plt.scatter(grid[:, 0], grid[:, 1])
    plt.savefig('results/%s' % filename)


def plot_grid_3d(grid, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2])
    plt.savefig('results/%s' % filename)


def plot_vectorfield_2d(grid, V, filename):
    plt.clf()
    plt.quiver(grid[:, 0], grid[:, 1], V[:, 0], V[:, 1])
    plt.savefig('results/%s' % filename)


def plot_vectorfield_3d(grid, V, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(grid[:, 0], grid[:, 1], grid[:, 2], V[:, 0], V[:, 1], V[:, 2])
    plt.savefig('results/%s' % filename)


def apply_transformation(image, transformation, dim=2, res=100):
    if dim == 2:
        grid = vector_fields.get_control_points_2d(image, res)
        x, y = image.shape
        grid_x, grid_y = np.mgrid[0:x:1, 0:y:1]
        full_grid = np.array((grid_y.flatten(), grid_x.flatten())).T
        #full_grid = vector_fields.get_control_points_2d(image, 1)
    if dim == 3:
        grid = vector_fields.get_control_points_3d(image, res)
        full_grid = vector_fields.get_control_points_3d(image, 1)
    grid_dense = forward_euler.interpolate_n_d(transformation, grid, full_grid).astype('float32')
    print(grid_dense.shape)
    warped = cv2.remap(image, grid_dense[:,0].reshape(image.shape),
                       grid_dense[:,1].reshape(image.shape), interpolation=cv2.INTER_CUBIC)
    return warped
