from skimage import io, color
import numpy as np
import cv2
import vector_fields
import forward_euler
import utils
import os


# An example workflow.
def main():
    example = io.imread('kanelsnurrer.jpg')
    example_2d = color.rgb2gray(example)
    example_3d_mockup = np.repeat(example, 200, 2)

    # get grid for kernels
    kernel_res = 100
    kernel_grid_2d = vector_fields.get_points_2d(example_2d, kernel_res)
    kernel_grid_3d = vector_fields.get_points_3d(example_3d_mockup, kernel_res)
    utils.plot_grid_2d(kernel_grid_2d, 'grid_2d.png')
    utils.plot_grid_3d(kernel_grid_3d, 'grid_3d.png')
    # get points for evaluation
    eval_points_res = 2
    eval_points_res = 2
    evaluation_points_2d = vector_fields.get_points_2d(example_2d, eval_points_res)
    evaluation_points_3d = vector_fields.get_points_3d(example_3d_mockup, eval_points_res)

    # Compute Gram/Evaluation matrix
    #S2d = vector_fields.gram_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, 400), grid_2d)
    #S3d = vector_fields.gram_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, 400), grid_3d)
    c_sup = 200
    if os.path.exists('evaluation_matrices/example2D_100_200_2.npy'):
        print('Loading evaluation matrix')
        S2d = utils.load_matrix('example2D_100_200_2.npy')
    else:
        print('Computing evaluation matrix')
        S2d = vector_fields.evaluation_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, c_sup), kernel_grid_2d,
                                              evaluation_points_2d)
    utils.save_matrix(S2d, 'example2D_100_200_2.npy')
    print('Done.')

    # Compute and plot vector fields
    d = 2
    V_2d = vector_fields.make_random_V(S2d, d)
    #V_3d = vector_fields.make_random_V(S3d, 3)

    utils.plot_vectorfield_2d(evaluation_points_2d, V_2d, 'vectorfield_2d.png') # this is too many points at this point - fix!
    #utils.plot_vectorfield_3d(grid_3d, V_3d, 'vectorfield_3d.png')

    x_10_2d = forward_euler.integrate(evaluation_points_2d, V_2d, 10)
    #x_10_3d = forward_euler.integrate(grid_3d, V_3d, 10)

    utils.plot_grid_2d(x_10_2d, 'transformation_2d.png')
    #utils.plot_grid_3d(x_10_3d, 'transformation_3d.png')

    warped = utils.apply_transformation(example_2d, x_10_2d, dim=d, res=2)
    cv2.imwrite('results/warped_2d.png', warped*255)


if __name__ == main():
    main()
