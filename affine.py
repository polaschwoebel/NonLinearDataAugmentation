import numpy as np
import vector_fields
import math as m
import registration
from skimage import io


def makeM(theta, scale, tx, ty, centerx, centery):
    alpha = scale * m.cos(theta)
    beta = scale * m.sin(theta)
    M = np.array([[alpha, beta, tx],
                   [-beta, alpha, ty]])
    return M


def make_phi(image, parameters):
    centery = image.shape[0] // 2
    centerx = image.shape[1] // 2

    theta, scale, tx, ty = parameters
    M = makeM(theta, scale, tx, ty, centerx, centery)

    points = vector_fields.get_points_2d(image, 1)
    centered_points = points - [centerx, centery]

    nr_points = points.shape[0]
    centered_homg_points = np.hstack([centered_points, np.ones((nr_points, 1))])

    centered_phi = M.dot(centered_homg_points.T).T
    phi = centered_phi + [centerx, centery]

    return phi

def plot_trafo(image, parameters):
    image = image.astype(np.float32)
    points = vector_fields.get_points_2d(image, 1)
    phi = make_phi(image, parameters)
    reconstructed = registration.apply_transformation(image,  points, phi, 1, return_image=True)
    io.imshow(reconstructed, cmap='gray')
    return reconstructed

######################## gradient ##################################

def dphi_dM_pointwise(phi, theta, scale, tx, ty):
    x = phi[0]
    y = phi[1]
    dphi_dM = np.zeros((2,4))
    # derivative w.r.t. theta
    dphi_dM[0, 0] = -scale*m.sin(theta)*x + scale*m.cos(theta)*y
    dphi_dM[1,0] = -scale*m.cos(theta)*x - scale*m.sin(theta)*y
    # scale
    dphi_dM[0,1] = m.cos(theta)*x + m.sin(theta)*y
    dphi_dM[1,1] = - m.sin(theta)*x +  m.cos(theta)*y
    # tx
    dphi_dM[0,2] = 1
    dphi_dM[1,2] = 0
    # ty
    dphi_dM[0,3] = 0
    dphi_dM[1,3] = 1
    return dphi_dM


def dphi_dM(image, parameters):
    theta, scale, tx, ty = parameters
    phi = make_phi(image, parameters)
    nr_points = phi.shape[0]
    dphi_dM = np.zeros((2*nr_points, 4))
    for i in range(nr_points):
        ix = i*2
        dphi_dM[ix:ix+2, :] = dphi_dM_pointwise(phi[i, :], theta, scale, tx, ty)
    return dphi_dM
