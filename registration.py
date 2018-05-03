import numpy as np
import vector_fields
import forward_euler
import utils
import gradient
from scipy import optimize


def apply_transformation(img_source, points, trafo, points_res, kernel_res=50, return_image=False):
    dim = len(img_source.shape)
    transformed_source_points = utils.interpolate_image(img_source, trafo, points_res)

    # bring points back into image shape - only for visualization
    if return_image:
        new_shape = utils.reconstruct_dimensions(img_source, points_res)
        print('new_shape:', new_shape)
        #if dim == 2:
        #    transformed_source_image = transformed_source_points.reshape(
        #        new_shape[0], new_shape[1], order='F')
        #if dim == 3:
        #    transformed_source_image = transformed_source_points.reshape(
        #            new_shape[0], new_shape[1], new_shape[2], order='F')
        #return transformed_source_image
        transformed_source_image = transformed_source_points.reshape(new_shape, order='F')
        return transformed_source_image

    else:
        return transformed_source_points


def E_D(img_source, img_target, points, trafo, debug=False):
    dim = len(img_source.shape)
    # note that we are actually applying the inverse transform,
    res = 50
    source_points = apply_transformation(img_source,  points, trafo, res, return_image=False)
    if dim == 2:
        target_points = img_target[points[:, 1], points[:, 0]]
    if dim == 3:
        target_points = img_target[points[:, 1], points[:, 0], points[:, 2]]
    return np.linalg.norm(source_points-target_points)**2


def E_R(G, alpha, sigma=1):
    print('alpha:', alpha.shape, 'G:', G.shape)
    reg_norm = alpha.T.dot(G.dot(alpha))
    weight = 0.05 # Akshay's suggestion
    return weight * reg_norm


def compute_error_and_gradient(im1, im2, points, kernels, alpha, S, kernel_res=100, eval_res=50, c_sup=200, dim=3):
        print('Compute transformation.')
        phi_1, dphi_dalpha_1 = forward_euler.integrate(points, kernels, alpha, S, steps=10)
        #phi_1 = np.rint(phi_1).astype(int)

        # recompute G
        #G = utils.get_G_from_S(S, kernel_res, eval_res, im1.shape)
        G = vector_fields.evaluation_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, c_sup), kernels,
                                            kernels, dim=dim, c_sup=c_sup)

        print('Compute Error.')
        E_Data = E_D(im1, im2, points, phi_1)
        print('Data term:', E_Data)
        E_Reg = E_R(G, alpha)
        print('Regularization term:', E_Reg, 'Done.')
        E = E_Data + E_Reg

        # data term error gradient
        dIm_dphi1 = gradient.dIm_dphi(im1, phi_1, eval_res)
        dED_dphi1 = gradient.dED_dphit(im1, im2, phi_1, points, dIm_dphi1)
        dER_dalpha = gradient.dER_dalpha(G, alpha)
        # final gradient
        final_gradient = gradient.error_gradient(dED_dphi1, dphi_dalpha_1, dER_dalpha)
        print('Error gradient:', final_gradient, np.where(final_gradient<0))
        return E, final_gradient


# final registration function for 3d
def find_transformation(im1, im2, kernel_res=100, eval_res=50, c_sup=200, d=3):
    print('Getting grid points.')
    if d==2:
        kernel_grid = vector_fields.get_points_2d(im1, kernel_res)
        points = vector_fields.get_points_2d(im1, eval_res)
    else:
        kernel_grid = vector_fields.get_points_3d(im1, kernel_res)
        points = vector_fields.get_points_3d(im1, eval_res)

    S = vector_fields.evaluation_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, c_sup), kernel_grid,
                                        points, dim=d)
    nxd = S.shape[1]
    alpha_0 = np.zeros(nxd)

    # optimization
    objective_function = (lambda alpha:
                          compute_error_and_gradient(im1, im2,
                                                     points, kernel_grid, alpha, S, kernel_res=kernel_res, eval_res=eval_res, c_sup=c_sup, dim=d))
    best_alpha = optimize.minimize(objective_function, alpha_0, jac=True, options={'disp':True})

    # some other optimizers to play with:
    #minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
    #best_alpha = optimize.basinhopping(objective_function, alpha_0, minimizer_kwargs=minimizer_kwargs,
    #                                   niter=200)
    # best_alpha = optimize.fmin_l_bfgs_b(objective_function, alpha_0)

    return best_alpha
