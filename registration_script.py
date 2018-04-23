from skimage import io
import numpy as np
import registration
import time


def main():
    start_time = time.time()
    example = io.imread('kanelsnurrer.jpg')
    example_3d_mockup = np.repeat(example, 101, 2)
    alpha = registration.find_transformation(example_3d_mockup, example_3d_mockup)
    print('Done. Found alpha:', alpha, "--- %s seconds ---" % (time.time() - start_time))


if __name__ == main():
    main()
