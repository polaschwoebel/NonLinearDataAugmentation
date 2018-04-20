import numpy as np
import vector_fields
import forward_euler
import utils
import os


def E_D(img_source, img_target, points, trafo, res, debug=False):
    # 'nearest neighbor interpolating' trafo to get integer indices
    trafo = np.rint(trafo).astype(int)
    dim = len(img_source.shape)
    # analogously to the cv2 function we are actually applying the INVERSE transform,
    if dim == 2:
        source_points = img_source[trafo[:, 1], trafo[:, 0]]
        target_points = img_target[points[:, 1], points[:, 0]]
    if dim == 3:
        source_points = img_source[trafo[:, 1], trafo[:, 0], trafo[:, 2]]
        target_points = img_target[points[:, 1], points[:, 0], points[:, 2]]
    return np.linalg.norm(source_points-target_points)**2


def E_R(S, alpha, img_shape, kernel_res, eval_res, sigma=1):
    G = utils.get_G_from_S(S, kernel_res, eval_res, img_shape)
    reg_norm = alpha.T.dot(G.dot(alpha))
    #log_likelihood = -norm/(2*sigma**2) + math.log(1/(sigma*math.sqrt(2*math.pi)))
    return reg_norm


# final registration function for 3d
def find_transformation(im1, im2):
    # choose parameters (could be inputs)
    kernel_res = 100
    eval_res = 50
    c_sup = 200

    print('Getting grid points.')
    kernel_grid = vector_fields.get_points_3d(im1, kernel_res)
    evaluation_points = vector_fields.get_points_3d(im1, eval_res)

    # S can be loaded if pre-computed, otherwise compute here
    if os.path.exists('evaluation_matrices/example3D_100_200_50.npy'):
        print('Loading evaluation matrix.')
        S = utils.load_matrix('example3D_100_200_50.npy')
    else:
        print('Computing S.')
        S = vector_fields.evaluation_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, c_sup), kernel_grid,
                                            evaluation_points)
        utils.save_matrix(S, 'example3D_100_200_2.npy')

    # HMCS
    nxd = S.shape[1]
    print('Initialize with zero vector field.')
    alpha = np.zeros(nxd)
    V = vector_fields.make_V(S, alpha, 3)
    print('Compute transformation.')
    phi_1, du_dalpha_1 = forward_euler.integrate(evaluation_points, V, 10)
    print('Compute likelihood.')
    E_Data = E_D(im1, im2, phi_1, evaluation_points)
    print('E_D:', E_D, '. Now compute prior.')
    E_Reg = E_R(S, alpha, im1.shape, kernel_res, eval_res)
    print('E_R:', E_R, 'Done.')
    E = E_Data + E_Reg


    # TODO:
    # -compute gradient w.r.t. alpha
    # -sample new trafo
    # - return alpha/trafo after some threshold is passed (iterations/error small?)
    return alpha, E
