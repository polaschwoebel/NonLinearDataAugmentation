import registration
from scipy import sparse
import numpy as np
import forward_euler
import vector_fields
import matlab.engine
import matlab
import time

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


def interpolate_image(image, eng, spline_rep, phi, res, dim):
    if (dim == 2):
        phi_x = matlab.double(phi[:,0].tolist())
        phi_y = matlab.double(phi[:,1].tolist())
        imres = image.shape[0]
        interpolation = np.array(eng.eval_fun2d(spline_rep, phi_x, phi_y, imres))
        # Set zeros where NaN
        interpolation[np.isnan(interpolation)] = 0
        return interpolation
    else:
        phi_x = matlab.double(phi[:,0].tolist())
        phi_y = matlab.double(phi[:,1].tolist())
        phi_z = matlab.double(phi[:,2].tolist())
        (xres, yres, zres) = image.shape
        interpolation = np.array(eng.eval_fun3d(spline_rep, phi_x, phi_y, phi_z,
                                                xres, yres, zres))
        # Set zeros where NaN
        interpolation[np.isnan(interpolation)] = 0
        return interpolation

# Apply transformation image at full resolution
def apply_trafo_full(im1, alpha, options):
    start = time.time()
    eng = matlab.engine.start_matlab()
    img_mat = matlab.double(im1.tolist())
    spline_rep = eng.BSrep(img_mat, options["dim"])
    print("APPLY -- started matlab ", (time.time() - start) / 60)
    start = time.time()
    if (options["dim"] == 2):
        points = vector_fields.get_points_2d(im1, 1)
        kernels = vector_fields.get_points_2d(im1, options['kernel_res'])
    else:
        points = vector_fields.get_points_3d(im1, 1)
        kernels = vector_fields.get_points_3d(im1, options['kernel_res'])
        points_filt = registration.filter_irrelevant_points(points, options["eval_mask"])
        kernels_filt = registration.filter_irrelevant_points(kernels, options["kernel_mask"])
    print("APPLY -- points.shape ", points.shape)
    print("APPLY -- points_filt.shape", points_filt.shape)
    print("APPLY -- prepared kernels and points ", (time.time() - start) / 60)
    start = time.time()
    phi = forward_euler.integrate(points_filt, kernels_filt, alpha, options['c_sup'], 
                                 options['dim'], steps=5, 
                                 compute_gradient = False)
    print("points.shape ", points.shape)
    print("phi.shape ", phi.shape)
    print("mask shape ", options["eval_mask"].shape)
    print("APPLY -- Forward Euler complete ", (time.time() - start) / 60)
    start = time.time()
#    points[options["eval_mask"].flatten() == 1] = phi
    mask_reshaped = np.concatenate(tuple([options["eval_mask"][:,:,i] for i
                                          in range(im1.shape[2])])).flatten()
    points[mask_reshaped == 1] = phi
    print("APPLY -- phi inserted into points ", (time.time() - start) / 60)
    start = time.time()
    image = interpolate_image(im1, eng, spline_rep, points, 
                              options['eval_res'], options["dim"])
    print("APPLY -- Applying inverse complete ", (time.time() - start) / 60)
    return image
    #return image.reshape(im1.shape, order='F')

def apply_trafo_full_in_chunks(im1, alpha, options):
    start = time.time()
    eng = matlab.engine.start_matlab()
    img_mat = matlab.double(im1.tolist())
    spline_rep = eng.BSrep(img_mat, options["dim"])
    print("APPLY -- started matlab ", (time.time() - start) / 60)
    start = time.time()
    if (options["dim"] == 2):
        points = vector_fields.get_points_2d(im1, 1)
        kernels = vector_fields.get_points_2d(im1, options['kernel_res'])
    else:
        points = vector_fields.get_points_3d(im1, 1)
        kernels = vector_fields.get_points_3d(im1, options['kernel_res'])
        points_filt = registration.filter_irrelevant_points(points, options["eval_mask"])
        kernels_filt = registration.filter_irrelevant_points(kernels, options["kernel_mask"])
#    print("APPLY -- points.shape ", points.shape)
#    print("APPLY -- points_filt.shape", points_filt.shape)
#    print("APPLY -- prepared kernels and points ", (time.time() - start) / 60)
    n_chunks = 50
    interval_size = points_filt.shape[0] // n_chunks
    phi_full = np.zeros_like(points_filt)
    for i in range(n_chunks):
        points0 = points_filt[i * interval_size:(i+1) * interval_size, :]
        start = time.time()
        phi = forward_euler.integrate(points0, kernels_filt, alpha, options['c_sup'], 
                                 options['dim'], steps=5, 
                                 compute_gradient = False)
        phi_full[i * interval_size:(i+1) * interval_size, :] = phi
        print("APPLY -- Forward Euler chunk " + str(i) + " ", (time.time() - start) / 60)
    start = time.time()
    mask_reshaped = np.concatenate(tuple([options["eval_mask"][:,:,i] for i
                                          in range(im1.shape[2])])).flatten()
    points[mask_reshaped == 1] = phi

    print("APPLY -- phi inserted into points ", (time.time() - start) / 60)
    start = time.time()
    image = interpolate_image(im1, eng, spline_rep, points, 
                              options['eval_res'], options["dim"])
    print("APPLY -- Applying inverse complete ", (time.time() - start) / 60)
    return image.reshape(im1.shape, order="F")
    #return image.reshape(im1.shape, order='F')


