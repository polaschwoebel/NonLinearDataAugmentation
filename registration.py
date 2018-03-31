import numpy as np
import vector_fields
import forward_euler
import utils
import os
import math


def apply_and_evaluate_transformation(img_source, img_target, points,
                                      trafo, res, debug=False):
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


def likelihood(im1, im2, trafo, evaluation_points, sigma=1):
    error = apply_and_evaluate_transformation(im1, im2, evaluation_points,
                                              trafo, res=2)
    print(-error/(2*sigma**2), )
    log_likelihood = -error/(2*sigma**2) + math.log(1/(sigma*math.sqrt(2*math.pi)))
    return log_likelihood


def prior(S, alpha, img_shape, kernel_res, eval_res, sigma=1):
    # find indices to recover gram matrix from large evaluation matrix
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
    S = S[indices, :]
    # actual computation of the prior
    norm = alpha.T.dot(S.dot(alpha))
    log_likelihood = -norm/(2*sigma**2) + math.log(1/(sigma*math.sqrt(2*math.pi)))
    return log_likelihood, S


# final registration function for 3d
def find_transformation(im1, im2):
    # choose parameters (could be inputs)
    kernel_res = 100
    eval_res = 50
    c_sup = 200

    print('Getting grid points.')
    kernel_grid = vector_fields.get_points_3d(im1, kernel_res)
    evaluation_points = vector_fields.get_points_3d(im1, eval_res)

    if os.path.exists('evaluation_matrices/example3D_100_200_50.npy'):
        print('Loading evaluation matrix.')
        S = utils.load_matrix('example3D_100_200_50.npy')
    else:
        print('Computing S.') # kernel matrix can alternatively be loaded if pre-computed
        S = vector_fields.evaluation_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, c_sup), kernel_grid,
                                            evaluation_points)
        utils.save_matrix(S, 'example3D_100_200_2.npy')

    # HMCS
    nxd = S.shape[1]
    print('Initialize with zero vector field.')
    alpha = np.zeros(nxd)
    V = vector_fields.make_V(S, alpha, 3)
    print('Compute transformation.')
    trafo = forward_euler.integrate(evaluation_points, V, 10)
    print('Compute likelihood.')
    log_likelihood = likelihood(im1, im2, trafo, evaluation_points)
    print('log_likelihood:', log_likelihood, '. Now compute prior.')
    log_prior = prior(S, alpha, im1.shape, kernel_res, eval_res)
    print('log_prior:', log_prior, 'Done.')
    log_posterior = log_likelihood + log_prior

    # TODO:
    # -compute gradient of the last line w.r.t. alpha
    # -sample new trafo
    # - return alpha/trafo after some threshold is passed (iterations/error small?)
    return alpha, log_posterior
