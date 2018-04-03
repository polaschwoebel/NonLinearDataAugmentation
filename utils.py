import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cv2
import numpy as np
import vector_fields
import forward_euler
import utils
from scipy import sparse


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


def enforce_boundaries(coords, img_shape):
    # make sure we are inside the image
    coords[:, 1] = coords[:, 1].clip(0, img_shape[1])
    coords[:, 0] = coords[:, 0].clip(0, img_shape[0])
    # 3d case
    if len(img_shape) == 3:
        coords[:, 2] = coords[:, 2].clip(0, img_shape[2])
    return coords


def save_matrix(matrix, file_name):
    sparse.save_npz('evaluation_matrices/%s' % file_name, matrix)


def load_matrix(file_name):
    return sparse.load_npz('evaluation_matrices/%s' % file_name)


def reconstruct_dimensions(image, res):
    d = len(image.shape)
    new_shape = []
    for dim in range(d):
        if image.shape[dim] % res == 0:
            new_shape.append(image.shape[dim]//res)
        else:
            new_shape.append(image.shape[dim]//res + 1)
    return new_shape


def apply_transformation(image, transformation, dim=2, res=2):
    if dim == 2:
        grid = vector_fields.get_points_2d(image, res)
        full_grid = vector_fields.get_points_2d(image, 1)
        grid_dense = forward_euler.interpolate_n_d(grid, transformation, full_grid).astype('float32')
        warped = cv2.remap(image, grid_dense[:, 0].reshape(image.shape, order='F'),
                           grid_dense[:, 1].reshape(image.shape, order='F'),
                           interpolation=cv2.INTER_NEAREST)
    if dim == 3:
        print('Not implemented for 3d, use registration.apply_and_evaluate_transformation() instead.')
        return
    return warped


def apply_and_evaluate_transformation_visual(img_source, img_target, points,
                                             trafo, res, debug=False):
    # 'nearest neighbor interpolating' trafo to get integer indices
    trafo = np.rint(trafo).astype(int)
    dim = len(img_source.shape)
    new_shape = utils.reconstruct_dimensions(img_source, res)
    # analogously to the cv2 function we are actually applying the INVERSE transform,
    # but this is confusing. change?
    if dim == 2:
        source_points = img_source[trafo[:, 1], trafo[:, 0]].reshape(
                    new_shape[0], new_shape[1], order='F')
        target_points = img_target[points[:, 1], points[:, 0]].reshape(
                   new_shape[0], new_shape[1], order='F')
        if debug:
            return (source_points, np.linalg.norm(source_points-target_points, 'fro')**2,
                    (source_points-target_points) != 0)
        else:
            return np.linalg.norm(source_points-target_points, 'fro')**2

    if dim == 3:
        source_points = img_source[trafo[:, 1], trafo[:, 0], trafo[:, 2]].reshape(
                    new_shape[0], new_shape[1], new_shape[2], order='F')
        target_points = img_target[points[:, 1], points[:, 0], points[:, 2]].reshape(
                       new_shape[0], new_shape[1], new_shape[2], order='F')
        if debug:
            return source_points, sum(sum(sum((source_points-target_points)**2))), (source_points-target_points)!=0
        else:
            return sum(sum(sum(source_points-target_points**2)))
