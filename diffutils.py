from scipy import sparse, ndimage
import numpy as np
import registration
import forward_euler
import vector_fields


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


def interpolate_image(image, phi_1, res):
    dim = phi_1.shape[-1]
    if dim == 2:
        coords = [phi_1[:, 1], phi_1[:, 0]]
    if dim == 3:
        coords = [phi_1[:, 1], phi_1[:, 0], phi_1[:, 2]]
    interpolated = ndimage.map_coordinates(image, coords, order = 1,
                                           mode='nearest')
    return interpolated


# recover gram matrix G from large evaluation matrix S by slicing
def get_G_from_S(S, kernel_res, eval_res, img_shape):
    d = 3
    resratio = kernel_res//eval_res
    eval_x_dim = img_shape[0]//eval_res + 1
    eval_y_dim = img_shape[1]//eval_res + 1
    eval_z_dim = img_shape[2]//eval_res + 1
    lowresrows = np.array([range(d*i*eval_y_dim*eval_x_dim, d*(i+1)*eval_y_dim*eval_x_dim)
                          for i in range(0, eval_z_dim, resratio)]).flatten()
    midresrows = np.array([range(d*i*eval_x_dim, d*(i+1)*eval_x_dim)
                          for i in range(0, eval_z_dim*eval_y_dim, resratio)]).flatten()
    highresrows = np.array([range(d*i, d*(i+1))
                           for i in range(0, eval_x_dim*eval_z_dim*eval_y_dim, resratio)]).flatten()
    keep = list(set(lowresrows) & set(midresrows) & set(highresrows))
    indices = np.zeros(S.shape[0])
    indices[keep] = 1
    indices = indices.astype(bool)
    G = S[indices, :]
    return G

# Apply transformation image at full resolution
def apply_trafo_full(I1, alpha, kernels, c_sup, dim):
    points = vector_fields.get_points_2d(I1, 1)
    S = vector_fields.evaluation_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, c_sup), 
                                        kernels, points, c_sup, dim=dim)
    phi, _ = forward_euler.integrate(points, kernels, alpha, S, c_sup, steps=10)
    return registration.apply_transformation(I1, points, phi, 1).reshape(I1.shape[0], I1.shape[1])
