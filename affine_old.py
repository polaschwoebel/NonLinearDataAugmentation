import numpy as np
import vector_fields
import registration
from skimage import io
from scipy import interpolate, sparse, optimize
import gradient


def make_M(affine_parameters):
    a1, a2, a3, a4 = affine_parameters
    M = np.array([[a1, a2],
                  [a3, a4]])
    return M


def translate(phi, translation_parameters, centerx, centery):
    tx, ty = translation_parameters
    tx = tx*centerx
    ty = ty*centery
    phi = phi + [tx, ty]
    return phi


def make_phi(image, affine_parameters, translation_parameters):
    centery = image.shape[0] // 2
    centerx = image.shape[1] // 2

    M = make_M(affine_parameters)
    points = vector_fields.get_points_2d(image, 1)
    centered_points = points - [centerx, centery]

    centered_phi = M.dot(centered_points.T).T
    affine_phi = centered_phi + [centerx, centery]
    phi = translate(affine_phi, translation_parameters, centerx, centery)
    return phi


def apply_trafo(image, affine_parameters, translation_parameters, eng, spline_rep, show_image=False):
    image = image.astype(np.float32)
    points = vector_fields.get_points_2d(image, 1)
    phi = make_phi(image, affine_parameters, translation_parameters)
    reconstructed = registration.apply_transformation(image, eng, spline_rep, points, phi, 1, return_image=True)
    if show_image:
        io.imshow(reconstructed, cmap='gray')
    return reconstructed


def dphi_dM_pointwise(phi, centerx, centery):
    x = phi[0]
    y = phi[1]
    dphi_dM = np.zeros((2,6))
    # derivative w.r.t. a1
    dphi_dM[0, 0] = x - centerx
    dphi_dM[1, 0] = 0
    # a2
    dphi_dM[0, 1] = y - centery
    dphi_dM[1, 1] = 0
    # a3
    dphi_dM[0, 2] = 0
    dphi_dM[1, 2] = x - centerx
    # a4
    dphi_dM[0, 3] = 0
    dphi_dM[1, 3] = y - centery
    # tx
    dphi_dM[0, 4] = centerx
    dphi_dM[1, 4] = 0
    # ty
    dphi_dM[0, 5] = 0
    dphi_dM[1, 5] = centery
    return dphi_dM


def dphi_dM(image, affine_parameters, translation_parameters):
    centery = image.shape[0] // 2
    centerx = image.shape[1] // 2
    phi = make_phi(image, affine_parameters, translation_parameters)
    nr_points = phi.shape[0]
    dphi_dM = np.zeros((2*nr_points, 6))
    for i in range(nr_points):
        ix = i*2
        dphi_dM[ix:ix+2, :] = dphi_dM_pointwise(phi[i, :], centerx, centery)
    return dphi_dM


def spline_dIm_dphi(img, phi):
    rows, columns = img.shape
    lx = np.linspace(0, columns-1, columns)
    ly = np.linspace(0, rows-1, rows)

    phi_x = phi[:, 0]
    phi_y = phi[:, 1]

    spline = interpolate.RectBivariateSpline(ly, lx,  # coords once only
                                             img.astype(np.float))
    x_grad = spline.ev(phi_y, phi_x, dx=1,
                       dy=0)

    y_grad = spline.ev(phi_y, phi_x, dx=0,
                       dy=1)
    all_grads = [x_grad, y_grad]

    gradient_array = np.dstack([dim_arr.flatten(order='F') for dim_arr in all_grads[::-1]])[0]
    block_diag = sparse.block_diag(gradient_array)
    return block_diag


def compute_error_and_gradient(im1, eng, spline_rep, im2, affine_parameters, translation_parameters, return_gradient=True):
    centery = im1.shape[0] // 2
    centerx = im1.shape[1] // 2

    points = vector_fields.get_points_2d(im1, 1)
    phi = make_phi(im1, affine_parameters, translation_parameters)

    E_data = registration.E_D(im1, eng, spline_rep, im2, points, phi, eval_res=1)
    if not return_gradient:
        return E_data
    dIm_dphi1 = gradient.dIm_dphi(im1, eng, spline_rep, phi, 1)
    dED_dphi1 = gradient.dED_dphit(im1, eng, spline_rep, im2, phi, points, dIm_dphi1, 1)
    dphi_dalpha = dphi_dM(im1, affine_parameters, translation_parameters)

    final_gradient = dED_dphi1.dot(dphi_dalpha).T

    return E_data, final_gradient.flatten()


def E_D_wrapper(im1, im2, parameters, eng, spline_rep):
    affine_parameters = parameters[:4]
    translation_parameters = parameters[4:]
    points = vector_fields.get_points_2d(im1, 1)
    phi = make_phi(im1, affine_parameters, translation_parameters)
    E_Data = registration.E_D(im1, eng, spline_rep, im2, points, phi, eval_res=1)
    return E_Data


def approximate_gradient(image, warped_image, bad_parameters, epsilon=1e-4):
    f_whole = lambda parameters: E_D_wrapper(image, warped_image, bad_parameters)
    approx_whole_gradient = optimize.approx_fprime(bad_parameters, f_whole, epsilon)
    return approx_whole_gradient


def find_transformation(im1, im2, eval_res, epsilon=1e-4):
    affine_parameters_0 = np.zeros(4)
    translation_parameters_0 = np.zeros(2)
    affine_parameters_0[0] = 1
    affine_parameters_0[4] = 1
    objective_function = (lambda affine_parameters, translation_parameters:
                          compute_error_and_gradient(im1, im2, affine_parameters, translation_parameters, return_gradient=False))
    best_alpha = optimize.minimize(objective_function, affine_parameters_0,translation_parameters_0,  jac=False, options={'disp':True, 'eps':epsilon})
    return best_alpha
