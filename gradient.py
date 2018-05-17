import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse
import diffutils as utils
import matplotlib.pyplot as plt
import time
#import cv2
import matlab.engine
import matlab

# This global variable is a bit hacky. It's 3 in the real registration, but could potentially be
# changed to 2 if we ever wanted to plot it.
dim = 2


# def dv_dphit(phi_t, kernels, alpha, c_sup=200): # D_phi ("spatial jacobian") in the paper
#     print('Compute spatial Jacobian.')
#     dim = kernels.shape[1]
#     alpha = alpha.reshape((-1,dim))
#     distances = euclidean_distances(phi_t, kernels)/c_sup
#     # m/3 is number of points, n/3 number of kernels
#     m, n = distances.shape
#     dv_dphit_diag = []
#     # computation of the sum of the kernel derivatives
#     # TODO: Rewrite following 2 loops as matrix multiplication?
#     for i in range(m):
#         phi_t_i = phi_t[i]
#         # get those kernel points where distance is smaller than 1, so kernel > 0
#         dist_smaller_1 = np.where(distances[i, :] < 1)[0]
#         alpha_indices = np.array([j for j in list(dist_smaller_1)])
#         sm = 0
#         # compute sum over all alpha+kernels that have a nonzero contribution
#         for alpha_idx in alpha_indices:
#             diff = phi_t_i - kernels[alpha_idx]
#             sm += alpha[alpha_idx].reshape((dim,1)).dot(diff.reshape((1,dim))) * (1 - np.linalg.norm(diff)/c_sup)**3
#         dv_dphit_i = -20/(c_sup**2) * sm
#         dv_dphit_diag.append(dv_dphit_i)
#     print('Done.')
#     return sparse.block_diag(dv_dphit_diag)


def dv_dphit(phi_t, kernels, alpha, c_sup): # D_phi ("spatial jacobian") in the paper
    #start = time.time()
    #alpha = alpha.reshape((-1, dim))
    alpha = alpha.reshape((-1, dim)).T

    distances = euclidean_distances(phi_t, kernels) / c_sup
    m, n = distances.shape
    dist_smaller_1 = np.where(distances < 1)

    diffs = np.zeros((kernels.shape[0], dim * phi_t.shape[0]))
    for i in range(len(dist_smaller_1[0])):
        m_c = dist_smaller_1[0][i]
        n_c = dist_smaller_1[1][i]
        diff = phi_t[m_c,:] - kernels[n_c,:]
        diffs[n_c,dim*m_c:dim*m_c+dim] = diff * (1-np.linalg.norm(diff) / c_sup) ** 3 * (-20/(c_sup**2))

    C = sparse.csc_matrix.dot(sparse.csc_matrix(alpha), sparse.csc_matrix(diffs))
    #print("GRADIENT - comp time: " + str(time.time() - start))
    #start = time.time()
    C = C.tolil()
    D = sparse.lil_matrix((dim * m, dim * m))
    for i in range(m):
        D[i*dim:(i+1)*dim,i*dim:(i+1)*dim] = C[:,i*dim:(i+1)*dim]
    #print("GRADIENT - make sparse: " + str(time.time() - start))
    return D


def next_dphi_dalpha(S, dv_dphit, prev_dphi_dalpha, step_size): # du_dalpha in the paper
    #print('GRADIENT -- dv_dphit:', dv_dphit.todense())
    m, n = S.shape
    p = step_size
    identity = sparse.identity(m)
    #print('GRADIENT -- shapes', (identity + 1/p * dv_dphit).shape, prev_dphi_dalpha.shape)

    dphi_dalpha = (identity + 1/p * dv_dphit).dot(prev_dphi_dalpha) + 1/p * S
    #print('GRADIENT -- dphi_dalpha shape. Supposed to be 2n x 2m.', dphi_dalpha.shape)
    #print('GRADIENT -- dphi_dalpha:', dphi_dalpha.todense())
    return dphi_dalpha

def dIm_dphi(img, eng, spline_rep, phi, res):
    #new_shape = utils.reconstruct_dimensions(img, res)
    #interpolation = utils.interpolate_image(img, eng, spline_rep, phi, res).reshape(new_shape, order='F')
    phi_x = matlab.double(phi[:,0].tolist())
    phi_y = matlab.double(phi[:,1].tolist())
    dev1 = np.array(eng.eval_dev1(spline_rep, phi_x, phi_y), dtype=np.float32)
    dev2 = np.array(eng.eval_dev2(spline_rep, phi_x, phi_y), dtype=np.float32)
    dev1[np.isnan(dev1)] = 0
    dev2[np.isnan(dev2)] = 0
    gradients_all_dims = [dev1, dev2]
    #gradients_all_dims = np.gradient(interpolation.astype(float))
    gradient_array = np.dstack([dim_arr.flatten(order='F') for dim_arr in gradients_all_dims])[::-1][0]
    block_diag = sparse.block_diag(gradient_array)
    return block_diag


def dIm_dphi_old(img, phi, res):
    new_shape = utils.reconstruct_dimensions(img, res)
    img_lowres = utils.interpolate_image(img, phi, res).reshape(new_shape, order='F')
    gradients_all_dims = np.gradient(img_lowres.astype(float))
    gradient_array = np.dstack([dim_arr.flatten(order='F') for dim_arr in gradients_all_dims])[0]
    # switch y and x here
    gradient_array = np.dstack([dim_arr.flatten(order='F') for dim_arr in gradients_all_dims[::-1]])[0]
    block_diag = sparse.block_diag(gradient_array)
    return block_diag


def dED_dphit(im1, eng, spline_rep, im2, phi_1, points, dIm1_dphi1, eval_res): #dEd_du in the paper
    #print('phi_1:', phi_1)
    source_points = utils.interpolate_image(im1, eng, spline_rep, phi_1, eval_res)
    #source_points = source_points.reshape(1, -1)
    #target_points = im2.reshape(1, -1)

    if dim==3:
        target_points = im2[points[:, 1], points[:, 0], points[:, 2]]
    else:
        target_points = im2[points[:, 1], points[:, 0]]
    diff =(source_points-target_points)
    #diff = sparse.csr_matrix(diff.reshape((-1,len(diff)), order='F'))
    diff = sparse.csr_matrix(diff)
    #print('GRADIENT-- check shapes:', diff.shape, dIm1_dphi1.shape)
    #return sparse.csc_matrix(2 * dIm1_dphi1.dot(full_dim_error))
    #print('GRADIENT -- point order?')
    #print('source_points:', source_points, 'target_points:', target_points)
    #print('diff:', diff.todense(), 'image gradient:', dIm1_dphi1.todense(), 'product:', diff.dot(dIm1_dphi1))
    #return
    #diff = sparse.csr_matrix(diff.reshape((-1,len(diff)), order='F'))
    return sparse.csc_matrix(2*diff.dot(dIm1_dphi1))


# remember that this is G not S!
def dER_dalpha(G, alpha):
    alpha = alpha.reshape((-1, 1))
    #print('GRADIENT - shapes dER_dalpha: supposed to be 3m x 1. G:', G.shape, 'alpha:', alpha.shape)
    return sparse.csc_matrix(2*G.dot(alpha))


# final gradient
def error_gradient(dED_dphit1, dphi_dalpha, dER_dalpha):
    weight = 0.05 # Akshay's suggestion
    data_gradient = dED_dphit1.dot(dphi_dalpha).T
    reg_gradient = dER_dalpha
    #print('GRADIENT -- shapes. E_D: supposed to be 2m x 1 (after transpose)', data_gradient.shape,
    #    'E_R: supposed to be 2m x 1', reg_gradient.shape)
    #print('GRADIENT -- data gradient:', data_gradient.todense())
    #print('GRADIENT -- reg. error:', reg_gradient.todense())
    gradient = data_gradient + weight*reg_gradient
    return gradient.toarray().flatten()
