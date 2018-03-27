import numpy as np


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
