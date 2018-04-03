from skimage import io, color
import numpy as np
import cv2
import registration

import os
import vector_fields
import forward_euler
import utils


# An example workflow.
def main():
    example = io.imread('brain.png')
    example_2d = color.rgb2gray(example)
    example_3d_mockup = np.repeat(example, 200, 2)

    # get grid for kernels
    kernel_res = 100
    kernel_grid_2d = vector_fields.get_points_2d(example_2d, kernel_res)
    kernel_grid_3d = vector_fields.get_points_3d(example_3d_mockup, kernel_res)
    utils.plot_grid_2d(kernel_grid_2d, 'grid_2d.png')
    utils.plot_grid_3d(kernel_grid_3d, 'grid_3d.png')
    # get points for evaluation. note: very low res in 3d for now due to computational feasibility
    eval_points_res_2d = 2
    eval_points_res_3d = 100
    evaluation_points_2d = vector_fields.get_points_2d(example_2d, eval_points_res_2d)
    evaluation_points_3d = vector_fields.get_points_3d(example_3d_mockup, eval_points_res_3d)

    # Compute Gram/Evaluation matrix 2d
    c_sup = 200
    if os.path.exists('evaluation_matrices/example2D_100_200_2.npy'):
        print('Loading evaluation matrix 2d')
        S2d = utils.load_matrix('example2D_100_200_2.npy')
    else:
        print('Computing evaluation matrix 2d')
        S2d = vector_fields.evaluation_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, c_sup), kernel_grid_2d,
                                              evaluation_points_2d)
    utils.save_matrix(S2d, 'example2D_100_200_2.npy')
    print('2d Done.')
    # Compute Gram/Evaluation matrix 3d
    if os.path.exists('evaluation_matrices/example3D_100_200_100.npy'):
        print('Loading evaluation matrix 3d')
        S3d = utils.load_matrix('example3D_100_200_100.npy')
    else:
        print('Computing evaluation matrix 3d')
        S3d = vector_fields.evaluation_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, c_sup), kernel_grid_3d,
                                              evaluation_points_3d)
    utils.save_matrix(S3d, 'example3D_100_200_100.npy')
    print('3d Done.')

    # Compute and plot vector fields
    d = 2
    V_2d = vector_fields.make_random_V(S2d, d)
    #V_3d = vector_fields.make_random_V(S3d, 3)

    utils.plot_vectorfield_2d(evaluation_points_2d, V_2d, 'vectorfield_2d.png') # this is too many points at this point - fix!
    #utils.plot_vectorfield_3d(evaluation_points_3d, V_3d, 'vectorfield_3d.png')

    x_10_2d = forward_euler.integrate(evaluation_points_2d, V_2d, 10)
    #x_10_3d = forward_euler.integrate(evaluation_points_3d, V_3d, 10)

    utils.plot_grid_2d(x_10_2d, 'transformation_2d.png')
    #utils.plot_grid_3d(x_10_3d, 'transformation_3d.png')

    warped_2d, error_2d, _ = utils.apply_and_evaluate_transformation_visual(example_2d, example_2d, evaluation_points_2d,
                                                                            x_10_2d, 2, debug=True)
    cv2.imwrite('results/warped_2d.png', warped_2d*255)
    print('2d done. Error is %s.' % error_2d)

    #error_3d = registration.apply_and_evaluate_transformation(example_3d_mockup, example_3d_mockup, evaluation_points_3d,
    #                                                          x_10_3d, 2)
    #print('3d done. Error is %s.' % error_3d)

if __name__ == main():
    main()
