from skimage import io, color
import numpy as np
import cv2
import os
import vector_fields
import forward_euler
import utils
import plotting_utils
import registration


# An example workflow.
def main():
    example = io.imread('brain.png')
    example_2d = color.rgb2gray(example)
    #example_3d_mockup = np.repeat(example, 101, 2)

    # get grid for kernels
    kernel_res = 100
    kernel_grid_2d = vector_fields.get_points_2d(example_2d, kernel_res)
    #kernel_grid_3d = vector_fields.get_points_3d(example_3d_mockup, kernel_res)
    plotting_utils.plot_grid_2d(kernel_grid_2d, 'kernel_grid_2d.png')
    #plotting_utils.plot_grid_3d(kernel_grid_3d, 'kernel_grid_3d.png')

    # get points for evaluation. note: low res in 3d for now due to computational feasibility
    eval_res_2d = 20
    #eval_res_3d = 50
    evaluation_points_2d = vector_fields.get_points_2d(example_2d, eval_res_2d)
    #evaluation_points_3d = vector_fields.get_points_3d(example_3d_mockup, eval_res_3d)

    # Compute Gram/Evaluation matrix 2d
    c_sup = 200
    if os.path.exists('evaluation_matrices/example2D_%s_%s_%s.npz' %(kernel_res, eval_res_2d, c_sup)):
        print('Loading evaluation matrix 2d')
        S2d = utils.load_matrix('example2D_%s_%s_%s.npz' %(kernel_res, eval_res_2d, c_sup))
    else:
        print('Computing evaluation matrix 2d')
        S2d = vector_fields.evaluation_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, c_sup), kernel_grid_2d,
                                              evaluation_points_2d)
        utils.save_matrix(S2d, 'example2D_%s_%s_%s.npz' %(kernel_res, eval_res_2d, c_sup))
    print('2d Done.')
    # Compute Gram/Evaluation matrix 3d
    # if os.path.exists('evaluation_matrices/example3D_%s_%s_%s.npz' %(kernel_res, eval_res_3d, c_sup)):
    #     print('Loading evaluation matrix 3d')
    #     S3d = utils.load_matrix('example3D_%s_%s_%s.npz' %(kernel_res, eval_res_3d, c_sup))
    # else:
    #     print('Computing evaluation matrix 3d')
    #     S3d = vector_fields.evaluation_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, c_sup), kernel_grid_3d,
    #                                           evaluation_points_3d)
    #     utils.save_matrix(S3d, 'example3D_%s_%s_%s.npz' %(kernel_res, eval_res_3d, c_sup))
    # print('3d Done.')

    # Compute and plot vector fields
    d = 2
    nxd = S2d.shape[1]
    alpha = (np.random.rand(nxd) - 0.5)*10
    V_2d = vector_fields.make_V(S2d, alpha, d)
    plotting_utils.plot_vectorfield_2d(evaluation_points_2d, V_2d, 'vectorfield_2d.png')

    x_1_2d = forward_euler.integrate(evaluation_points_2d, kernel_grid_2d, alpha, S2d, compute_gradient=False)
    plotting_utils.plot_grid_2d(x_1_2d, 'transformation_2d.png')
    print(example_2d.shape)
    warped_2d = registration.apply_transformation(example_2d, evaluation_points_2d, x_1_2d, res=eval_res_2d, return_image=True)
    # TODO: add 'dense application' that evaluates all points at the same time
    cv2.imwrite('results/warped_2d.png', warped_2d*255)
    print('2d plotting done.')

    # TODO : add a 3d visualization version if we want to; most functionality is already there (commented out)


if __name__ == main():
    main()
