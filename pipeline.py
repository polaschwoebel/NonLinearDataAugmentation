from skimage import io, color
import numpy as np
import cv2
import vector_fields
import forward_euler
import utils


# An example workflow.
def main():
    example = io.imread('kanelsnurrer.jpg')
    example_2d = color.rgb2gray(example)
    example_3d_mockup = np.repeat(example, 200, 2)

    grid_2d = vector_fields.get_control_points_2d(example_2d, 100)
    grid_3d = vector_fields.get_control_points_3d(example_3d_mockup, 100)
    utils.plot_grid_2d(grid_2d, 'grid_2d.png')
    utils.plot_grid_3d(grid_3d, 'grid_3d.png')

    S2d = vector_fields.gram_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, 400), grid_2d)
    S3d = vector_fields.gram_matrix(lambda x1, x2: vector_fields.kernel(x1, x2, 400), grid_3d)
    V_2d = vector_fields.make_random_V(S2d, 2)
    V_3d = vector_fields.make_random_V(S3d, 3)

    utils.plot_vectorfield_2d(grid_2d, V_2d, 'vectorfield_2d.png')
    utils.plot_vectorfield_3d(grid_3d, V_3d, 'vectorfield_3d.png')

    x_10_2d = forward_euler.integrate(grid_2d, V_2d, 10)
    x_10_3d = forward_euler.integrate(grid_3d, V_3d, 10)

    utils.plot_grid_2d(x_10_2d, 'transformation_2d.png')
    utils.plot_grid_3d(x_10_3d, 'transformation_3d.png')

    warped = utils.apply_transformation(example_2d, x_10_2d, dim=2, res=100)
    print(warped)
    cv2.imwrite('results/warped_2d.png', warped*255)


if __name__ == main():
    main()
