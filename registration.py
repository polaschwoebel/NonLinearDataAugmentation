import numpy as np
import vector_fields
import forward_euler
import diffutils as utils
import gradient
from scipy import optimize
import matlab.engine
import matlab


def apply_transformation(img_source, eng, spline_rep, points, trafo, points_res, return_image=False):
    dim = len(img_source.shape)
    transformed_source_points = utils.interpolate_image(img_source, eng, spline_rep, trafo, points_res)
    # bring points back into image shape - only for visualization
    if return_image:
        new_shape = utils.reconstruct_dimensions(img_source, points_res)
        transformed_source_image = transformed_source_points.reshape(new_shape, order='F')
        return transformed_source_image

    else:
        return transformed_source_points


def E_D(img_source, eng, spline_rep, img_target, points, trafo, debug=False, eval_res=1):
    dim = len(img_source.shape)
    # note that we are actually applying the inverse transform,
    source_points = apply_transformation(img_source, eng, spline_rep, points, trafo, eval_res, return_image=False)
    #target_points = img_target[points[:,0], points[:,1]]
    if dim == 2:
        target_points = img_target[points[:, 1], points[:, 0]]
    if dim == 3:
        target_points = img_target[points[:, 1], points[:, 0], points[:, 2]]
    return np.linalg.norm(source_points-target_points)**2


def E_R(G, alpha, sigma=1):
    reg_norm = alpha.T.dot(G.dot(alpha))
    return reg_norm


def compute_error_and_gradient(im1, eng, spline_rep, im2, points, kernels, 
                               alpha, kernel_res, eval_res, c_sup, dim, 
                               reg_weight):
        print(alpha.shape)
        print(kernels.shape)
        print(points.shape)
        phi_1, dphi_dalpha_1 = forward_euler.integrate(points, kernels, alpha, 
                                                       c_sup, dim, steps = 10)
        
        # Compute Gram matrix such that kernel_res = eval_res. Used for E_R
        G = vector_fields.evaluation_matrix(kernels, kernels, c_sup, dim)
        
        # Compute dissimilarity error
        E_Data = E_D(im1, eng, spline_rep, im2, points, phi_1, eval_res = eval_res)
        print('REG -- Data term:', E_Data)
        
        # Compute regularization error
        E_Reg = E_R(G, alpha)
        print('REG -- Regularization term:', E_Reg)
        E = E_Data + reg_weight * E_Reg
        
        ### GRADIENT ###
        dIm_dphi1 = gradient.dIm_dphi(im1, eng, spline_rep, phi_1, eval_res)

        dED_dphi1 = gradient.dED_dphit(im1, eng, spline_rep, im2, phi_1, points, dIm_dphi1, eval_res)
        dER_dalpha = gradient.dER_dalpha(G, alpha)
        final_gradient = gradient.error_gradient(dED_dphi1, dphi_dalpha_1, dER_dalpha)
        return E, final_gradient


# final registration function
def find_transformation(im1, im2, options):
    # Construct grid point and evaluation point structure
    if options["dim"] == 2:
        kernel_grid = vector_fields.get_points_2d(im1, options["kernel_res"])
        points = vector_fields.get_points_2d(im1, options["eval_res"])
    else:
        kernel_grid = vector_fields.get_points_3d(im1, options["kernel_res"])
        points = vector_fields.get_points_3d(im1, options["eval_res"])

    m3 = points.shape[0]
    n3 = kernel_grid.shape[0]
    
    # Initialize alpha
    alpha_0 = np.zeros(options["dim"] * n3)

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    # Convert image into suitable format for MATLAB
    img_mat = matlab.double(im1.tolist())
    # Create the spline representation using BSrep.m
    spline_rep = eng.BSrep(img_mat)

    # Optimization
    objective_function = (lambda alpha:
                          compute_error_and_gradient(im1, eng, spline_rep, im2,
                                                     points, kernel_grid, alpha, 
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
