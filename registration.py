import numpy as np
import vector_fields
import forward_euler
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


#def efficient_prior(S, alpha, kernel_res=100, eval_res=50, sigma=1): # or V directly for efficiency?
#    # here it will be neccessary to recover the right entries of the evaluation matrix!
#    print(S.shape)
#    print(kernel_res//eval_res)
#    rows = np.array([[i, i+1, i+2] for i in range(0, S.shape[0], (kernel_res//eval_res + 3)*3)]).flatten()
#    print(rows[0:10], rows[-10:])
#    S = S[rows, :]
#    print(S.shape, alpha.shape)
#    norm = alpha.T.dot(S.dot(alpha))
#    log_likelihood = -norm/(2*sigma**2) + math.log(1/(sigma*math.sqrt(2*math.pi)))
#    return log_likelihood


def prior(gram_matrix, alpha, kernel_res=100, eval_res=50, sigma=1): # or V directly for efficiency?
    norm = alpha.T.dot(gram_matrix.dot(alpha))
    log_likelihood = -norm/(2*sigma**2) + math.log(1/(sigma*math.sqrt(2*math.pi)))
    return log_likelihood


# final registration function for 3d
def find_transformation(im1, im2):
    # choose parameters (could be inputs)
    kernel_res = 100
    eval_res = 50
    c_sup = 200

    print('Getting grid points.')
    kernel_grid = vector_fields.get_points_3d(im1, kernel_res)
    evaluation_points = vector_fields.get_points_3d(im1, eval_res)
    # kernel matrix can alternatively be loaded if pre-computed
    print('Computing S.')
    S = vector_fields.evaluation_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, c_sup), kernel_grid,
                                        evaluation_points)
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
    # this is the easy and unefficient way to do it - see above
    print('Compute gram matrix.')
    gram_matrix = vector_fields.gram_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, c_sup), kernel_grid,)
    log_prior = prior(gram_matrix, alpha)
    print('log_prior:', log_prior, 'Done.')
    log_posterior = log_likelihood + log_prior

    # TODO:
    # -compute gradient of the last line w.r.t. alpha
    # -sample new trafo
    # - return alpha/trafo after some threshold is passed (iterations/error small?)
    return alpha, log_posterior
