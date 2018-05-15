from scipy import sparse, ndimage, interpolate
import numpy as np


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


def spline(img, phi, res, return_gradient=False):
    rows, columns = img.shape
    lx = np.linspace(0, columns-1, columns)
    ly = np.linspace(0, rows-1, rows)
    phi_x = phi[:,0]
    phi_y = phi[:,1]

    spline = interpolate.RectBivariateSpline(ly, lx, # coords once only
                                                img.astype(np.float))

    interpolated =  spline.ev(phi_y, phi_x, dx = 0,
              dy = 0)
    if not return_gradient:
        return interpolated
    x_grad = spline.ev(phi_y, phi_x, dx = 1,
                  dy = 0)

    y_grad = spline.ev(phi_y, phi_x, dx = 0,
                  dy = 1)
    all_grads = [x_grad, y_grad]
    gradient_array = np.dstack([dim_arr.flatten(order='F') for dim_arr in all_grads[::-1]])[0]
    block_diag = sparse.block_diag(gradient_array)
    return block_diag


def interpolate_image(image, phi_1, res):
    dim = phi_1.shape[-1]
    if dim == 2:
        coords = [phi_1[:, 1], phi_1[:, 0]]
    if dim == 3:
        coords = [phi_1[:, 1], phi_1[:, 0], phi_1[:, 2]]
    interpolated = ndimage.map_coordinates(image, coords, mode='nearest')
    #interpolated = spline(image, phi_1, res) # doesn't wprk?!?
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
