import numpy as np
import registration
import time


def main():
    example = np.load("kanelsnurrer.npy")
    example_3d_mockup = np.repeat(example, 101, 2)

    print('Trivial test: Same image twice - find identity transform.')
    start_time = time.time()
    alpha = registration.find_transformation(example_3d_mockup, example_3d_mockup)
    print('Done. Found alpha:', alpha, "\n --- %s seconds ---" % (time.time() - start_time), '\n')

    #TODO: Non-trivial example.
    # print('Test: Use random transformation and try to recover back.')

if __name__ == main():
    main()
