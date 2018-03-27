import numpy as np
import vector_fields
import forward_euler


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


# final registration function for 3d
def find_transformation(im1, im2):
    # choose parameters (could be inputs)
    kernel_res = 100
    eval_res = 50
    c_sup = 200

    kernel_grid = vector_fields.get_points_3d(im1, kernel_res)
    evaluation_points = vector_fields.get_points_3d(im1, eval_res)
    # kernel matrix can alternatively loaded if pre-computed
    S = vector_fields.evaluation_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, c_sup), kernel_grid,
                                        evaluation_points)

    # HMCS
    # initialization - will later be at 0 probably
    V = vector_fields.make_random_V(S, 3)
    x_10 = forward_euler.integrate(evaluation_points, V, 10)
    error = apply_and_evaluate_transformation(im1, im2, evaluation_points,
                                              x_10, 2)
    # TODO:
    # -compute gradient of the last line w.r.t. alpha
    # -sample new trafo
    # - return alpha/trafo after some threshold is passed (iterations/error small?)

    return
