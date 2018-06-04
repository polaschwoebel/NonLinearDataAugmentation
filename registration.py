import numpy as np
import matlab.engine
import matlab
from scipy import optimize

import vector_fields
import forward_euler
import diffutils as utils
import gradient

# Computation of the dissimilarity error
def E_D(im_source, im_target, trans_points, points, dim):
    if dim == 2:
        target_points = im_target[points[:, 1], points[:, 0]]
    if dim == 3:
        target_points = im_target[points[:, 1], points[:, 0], points[:, 2]]
    return np.linalg.norm(trans_points - target_points) ** 2

# Computation of the regularization error
def E_R(G, alpha):
    reg_norm = alpha.T.dot(G.dot(alpha))
    return reg_norm

def compute_error_and_gradient(im_source, eng, spline_rep, im_target, points,
                               kernels, alpha, kernel_res, eval_res, c_sup,
                               dim, reg_weight):
        # Compute the integration using the Forward Euler method along with gradient
        phi, dphi_dalpha = forward_euler.integrate(points, kernels, alpha,
                                                   c_sup, dim, steps=10)

        # Compute Gram matrix such that kernel_res = eval_res. Used for E_R
        G = vector_fields.evaluation_matrix(kernels, kernels, c_sup, dim)

        trans_points = utils.interpolate_image(im_source, eng, spline_rep, phi,
                                               eval_res, dim)

        # Compute dissimilarity error
        E_Data = E_D(im_source, im_target, trans_points, points, dim)
        print('REG -- Data term:', E_Data)

        # Compute regularization error
        E_Reg = E_R(G, alpha)
        print('REG -- Regularization term:', reg_weight * E_Reg)
        E = E_Data + reg_weight * E_Reg

        ### GRADIENT ###
        dIm_dphi1 = gradient.dIm_dphi(im_source, eng, spline_rep, phi, eval_res, dim)
        dED_dphi1 = gradient.dED_dphit(im_source, im_target, trans_points,
                                       points, dIm_dphi1, dim)
        dER_dalpha = gradient.dER_dalpha(G, alpha)

        data_gradient = dED_dphi1.dot(dphi_dalpha).T
        final_gradient = data_gradient + reg_weight * dER_dalpha
        final_gradient = final_gradient.toarray().flatten()
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
    # Construct grid point and evaluation point structure
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

    n = kernels.shape[0]

    # Initialize alpha
    alpha_0 = np.zeros(options["dim"] * n)

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    # Convert image into suitable format for MATLAB
    img_mat = matlab.double(im1.tolist())
    # Create the spline representation using BSrep.m
    spline_rep = eng.BSrep(img_mat, options["dim"])

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
