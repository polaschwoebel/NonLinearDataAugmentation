import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse
import utils

# This global variable is a bit hacky. It's  3 in the real registration, but could potentially be
# changed to 2 if we ever wanted to plot it.
dim = 3

def dv_dphit(phi_t, kernels, alpha, c_sup=200): # D_phi ("spatial jacobian") in the paper
    print('Compute spatial Jacobian.')
    dim = kernels.shape[1]
    alpha = alpha.reshape((-1,dim))
    distances = euclidean_distances(phi_t, kernels)/c_sup
    # m/3 is number of points, n/3 number of kernels
    m, n = distances.shape
    dv_dphit_diag = []
    # computation of the sum of the kernel derivatives
    # TODO: Rewrite following 2 loops as matrix multiplication?
    for i in range(m):
        phi_t_i = phi_t[i]
        # get those kernel points where distance is smaller than 1, so kernel > 0
        dist_smaller_1 = np.where(distances[i, :] < 1)[0]
        alpha_indices = np.array([j for j in list(dist_smaller_1)])
        sm = 0
        # compute sum over all alpha+kernels that have a nonzero contribution
        for alpha_idx in alpha_indices:
            diff = phi_t_i - kernels[alpha_idx]
            sm += alpha[alpha_idx].reshape((dim,1)).dot(diff.reshape((1,dim))) * (1 - np.linalg.norm(diff)/c_sup)**3
        dv_dphit_i = -20/c_sup**2 * sm
        dv_dphit_diag.append(dv_dphit_i)
    print('Done.')
    return sparse.block_diag(dv_dphit_diag)


def next_dphi_dalpha(S, dv_dphit, prev_dphi_dalpha, step_size): # du_dalpha in the paper
    m, n = S.shape
    p = step_size
    identity = sparse.identity(m)
    du_dalpha = (identity + 1/p * dv_dphit).dot(prev_dphi_dalpha) + 1/p * S
    return du_dalpha


def dIm_dphi(img, phi, res):
    new_shape = utils.reconstruct_dimensions(img, res)
    # TODO: interpolate instead of round
    phi = np.rint(phi).astype(int)
    img_lowres = img[[phi[:, 1], phi[:, 0], phi[:,2]]].reshape(
                       new_shape[0], new_shape[1], new_shape[2], order='F')
    img_lowres = utils.interpolate_image(img, phi, res).reshape(new_shape, order='F')

    gradients_3dims = np.gradient(img_lowres)
    gradient_3d_array = np.dstack([dim_arr.flatten() for dim_arr in gradients_3dims])
    gradient_diagonal = gradient_3d_array.flatten()
    return sparse.diags(gradient_diagonal)


def dED_dphit(im1, im2, phi_1, points, dIm1_dphi1, res=50): #dEd_du in the paper
    source_points = utils.interpolate_image(im1, phi_1, res)
    target_points = im2[points[:, 1], points[:, 0], points[:, 2]]
    full_dim_error = np.repeat(source_points-target_points, dim)
    return sparse.csc_matrix(2 * dIm1_dphi1.dot(full_dim_error))


# remember that this is G not S!
def dER_dalpha(G, alpha):
    #alpha = alpha.flatten()
    return sparse.csc_matrix(2*G.dot(alpha))


# final gradient
def error_gradient(dED_dphit1, dphi_dalpha, dER_dalpha):
    weight = 0.05 # Akshay's suggestion
    gradient = dED_dphit1.dot(dphi_dalpha) + weight*dER_dalpha
    return gradient.toarray().flatten()
