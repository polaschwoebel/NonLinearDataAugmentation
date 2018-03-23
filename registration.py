import numpy as np


def apply_and_evaluate_transformation(img_source, img_target, points,
                                      trafo, res, return_image=False):
    # 'nearest neighbor interpolating' trafo to get integer indices
    trafo = np.rint(trafo).astype(int)
    # analogously to the cv2 function we are actually applying the INVERSE transform,
    # but should be ok?
    source_points = img_source[trafo[:, 1], trafo[:, 0]].reshape(
                    img_source.shape[1]//res, img_source.shape[0]//res).T
    target_points = img_target[points[:, 1], points[:, 0]].reshape(
                    img_target.shape[1]//res, img_target.shape[0]//res).T
    return np.linalg.norm(source_points-target_points, 'fro')
