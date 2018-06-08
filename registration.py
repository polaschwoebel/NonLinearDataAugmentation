import matlab.engine
import matlab

import numpy as np
from scipy import optimize
import time

import vector_fields
import forward_euler
import diffutils as utils
import gradient

import time

# Computation of the dissimilarity error
def E_D(im_source, im_target, trans_points, points, dim):
    if dim == 2:
        target_points = im_target[points[:, 1], points[:, 0]]
    if dim == 3:
        target_points = im_target[points[:, 1], points[:, 0], points[:, 2]]
    return np.linalg.norm(trans_points - target_points) ** 2

# Computation of the regularization error
def E_R(G, alpha, dim):
    reg_norm = alpha.T.dot(G.dot(alpha))
    return reg_norm

def compute_error_and_gradient(im_source, eng, spline_rep, im_target, points,
                               kernels, alpha, kernel_res, eval_res, c_sup,
                               dim, reg_weight):
        start = time.time()
        # Compute the integration using the Forward Euler method along with gradient
        phi, dphi_dalpha = forward_euler.integrate(points, kernels, alpha,
                                                   c_sup, dim, steps=5)
        print("REG -- Integration: ", (time.time() - start) / 60)
       
        # Compute Gram matrix such that kernel_res = eval_res. Used for E_R
        G = vector_fields.evaluation_matrix_blowup(kernels, kernels, c_sup, dim)

        start = time.time()
        trans_points = utils.interpolate_image(im_source, eng, spline_rep, phi,
                                               eval_res, dim)
        print("REG -- I(phi)", (time.time() - start) / 60)

        # Compute dissimilarity error
        E_Data = E_D(im_source, im_target, trans_points, points, dim)
        print('**** DATA TERM **** ', E_Data)

        # Compute regularization error
        E_Reg = E_R(G, alpha, dim)
        print('**** REGULARIZATION TERM **** ', reg_weight * E_Reg)
        E = E_Data + reg_weight * E_Reg

        ### GRADIENT ###
        start = time.time()
        dIm_dphi1 = gradient.dIm_dphi(im_source, eng, spline_rep, phi, eval_res, dim)
        print('REG -- dIm_dphi1: ', (time.time() - start) / 60)

        start = time.time()

        dED_dphi1 = gradient.dED_dphit(im_source, im_target, trans_points,
                                       points, dIm_dphi1, dim)
        print('REG -- dE_dphi1: ', (time.time() - start) / 60)

        dER_dalpha = gradient.dER_dalpha(G, alpha)
        start = time.time()
        data_gradient = dED_dphi1.dot(dphi_dalpha).T
        final_gradient = data_gradient + reg_weight * dER_dalpha
        final_gradient = final_gradient.toarray().flatten()
        print('REG -- final_gradient: ', (time.time() - start) / 60)
        return E, final_gradient


def filter_irrelevant_points(points, mask):
    pl = points.tolist()
    pl_new = []
    for i in range(len(pl)):
        if (mask[tuple(pl[i])]):
            pl_new.append(pl[i])
    return np.array(pl_new)

# Find optimal alphas given 2 images using scipy optimizer
def find_transformation(im1, im2, options):
    print('REG -- start reg.')
    # Construct grid point and evaluation point structure
    start = time.time()
    if options["dim"] == 2:
        kernels = vector_fields.get_points_2d(im1, options["kernel_res"])
        points = vector_fields.get_points_2d(im1, options["eval_res"])
        kernels = filter_irrelevant_points(kernels, options["kernel_mask"])
        points = filter_irrelevant_points(points, options["eval_mask"])
    else:
        kernels = vector_fields.get_points_3d(im1, options["kernel_res"])
        points = vector_fields.get_points_3d(im1, options["eval_res"])
        kernels = filter_irrelevant_points(kernels, options["kernel_mask"])
        points = filter_irrelevant_points(points, options["eval_mask"])
    print("REG -- Constructed kernels and points: ", (time.time() - start) / 60)
    n = kernels.shape[0]
    print("KERNEL SIZE ", kernels.shape)
    print("EVAL SIZE ", points.shape)
    # Initialize alpha
    alpha_0 = np.zeros(options["dim"] * n)

    start = time.time()
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    print("REG -- started engine: ", (time.time() - start) / 60)
    start = time.time()

    # Convert image into suitable format for MATLAB
    img_mat = matlab.double(im1.tolist())
    print("REG -- made image into matlab doubles: ", (time.time() - start) / 60)

    # Create the spline representation using BSrep.m
    start = time.time()
    spline_rep = eng.BSrep(img_mat, options["dim"])
    print("REG -- Made spline rep: ", (time.time() - start) / 60)

    # Optimization
    objective_function = (lambda alpha:
                          compute_error_and_gradient(im1, eng, spline_rep, im2,
                                                     points, kernels, alpha,
                                                     options["kernel_res"],
                                                     options["eval_res"],
                                                     options["c_sup"],
                                                     options["dim"],
                                                     options["reg_weight"]))
    opt_res = optimize.minimize(objective_function, alpha_0, method = "BFGS",
                                jac = True,
                                options = {"disp" : True,
                                           "eps" : options["opt_eps"],
                                           "maxiter" : options["opt_maxiter"]})
    return opt_res["x"]
