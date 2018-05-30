import matplotlib.pyplot as plt
import math as m
from mpl_toolkits import mplot3d


def plot_grid_2d(grid, filename):
    plt.clf()
    plt.scatter(grid[:, 0], grid[:, 1])
    plt.savefig('results/%s' % filename)


def plot_grid_3d(grid, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2])
    plt.savefig('results/%s' % filename)


def plot_vectorfield_2d(grid, V, filename):
    plt.clf()
    plt.quiver(grid[:, 0], grid[:, 1], V[:, 0], V[:, 1])
    plt.savefig('results/%s' % filename)


def plot_vectorfield_3d(grid, V, filename):
    fig = plt.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(grid[:, 0], grid[:, 1], grid[:, 2], V[:, 0], V[:, 1], V[:, 2])
    plt.savefig('results/%s' % filename)

def plot_ims(ims, two_rows, align_zoom):
    plt.close("all")
    if two_rows:
        if align_zoom:
            f, spl = plt.subplots(2, int(m.ceil(len(ims) / 2)), sharex = True, sharey = True)
        else:
            f, spl = plt.subplots(2, int(m.ceil(len(ims) / 2)))
    else:
        if align_zoom:
            f, spl = plt.subplots(1, len(ims), sharex = True, sharey = True)
        else:
            f, spl = plt.subplots(1, len(ims))
    spl = spl.ravel()
    for i in range(len(ims)):
        spl[i].imshow(ims[i])