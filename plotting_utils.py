import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def plot_points_2d(grid, filename):
    plt.clf()
    plt.scatter(grid[:, 0], grid[:, 1], s=50)
    plt.savefig('results/%s' % filename)


def plot_grid_2d(image, grid, filename, with_kernels=False, kernels=None):
    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 5))
    if image is not None:
        ax.imshow(image, cmap='gray')
    length = 28
    eval_res = 2
    # plot vertical lines
    for column in range(length//eval_res):
        ax.plot(grid[column*14:(column*14) + 14, 0], grid[column*14:(column*14) + 14, 1], color='r')
    # plot horizontal lines
    for row in range(length):
        ax.plot(grid[row::14,0], grid[row::14, 1], color='r')
    if with_kernels:
            plt.scatter(kernels[:, 0], kernels[:, 1], s=50)
    plt.savefig('results/%s' % filename)

def plot_grid_3d(grid, filename):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2])
    plt.savefig('results/%s' % filename)


def plot_vectorfield_2d(grid, V, filename):
    plt.clf()
    fig, ax = plt.subplots(figsize=(5,5))
    plt.quiver(grid[:, 0], grid[:, 1], V[:, 0], V[:, 1])
    plt.savefig('results/%s' % filename)


def plot_vectorfield_3d(grid, V, filename):
    fig = plt.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(grid[:, 0], grid[:, 1], grid[:, 2], V[:, 0], V[:, 1], V[:, 2])
    plt.savefig('results/%s' % filename)

def plot_ims(ims):
    f, spl = plt.subplots(1, len(ims))
    spl = spl.ravel()
    for i in range(len(ims)):
        spl[i].imshow(ims[i])
