from scipy import sparse
import numpy as np
import forward_euler
import vector_fields
import matlab.engine
import matlab


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


def interpolate_image(image, eng, spline_rep, phi, res):
    phi_x = matlab.double(phi[:,0].tolist())
    phi_y = matlab.double(phi[:,1].tolist())
    imres = image.shape[0]
    interpolation = np.array(eng.eval_fun(spline_rep, phi_x, phi_y, imres))
    # Set zeros where NaN
    interpolation[np.isnan(interpolation)] = 0
    return interpolation


# Apply transformation image at full resolution
def apply_trafo_full(im1, alpha, options):
    eng = matlab.engine.start_matlab()
    img_mat = matlab.double(im1.tolist())
    spline_rep = eng.BSrep(img_mat)

    points = vector_fields.get_points_2d(im1, 1)
    kernels = vector_fields.get_points_2d(im1, options['kernel_res'])

    phi, _ = forward_euler.integrate(points, kernels, alpha, options['c_sup'], options['dim'], steps=10)
    return interpolate_image(im1, eng, spline_rep, phi, options['eval_res']).reshape(im1.shape, order='F')
