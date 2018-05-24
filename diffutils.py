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
    interpolation = np.array(eng.eval_fun(spline_rep, phi_x, phi_y))
    # Set zeros where NaN
    interpolation[np.isnan(interpolation)] = 0
    return interpolation

# Apply transformation image at full resolution
def apply_trafo_full(I1, alpha, kernels, c_sup, dim, eng, spline_rep, eval_res):
    points = vector_fields.get_points_2d(I1, 1)
    phi, _ = forward_euler.integrate(points, kernels, alpha, c_sup, dim, steps=10)
    return interpolate_image(I1, eng, spline_rep, phi, eval_res).reshape(I1.shape[0], I1.shape[1])
