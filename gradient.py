import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse


def dv_dphit(phi_t, kernels, alpha, c_sup=200):
    distances = euclidean_distances(phi_t, kernels)/c_sup
    # m/3 is number of points, n/3 number of kernels
    m, n = distances.shape
    dv_dphit_diag = []
    for i in range(m):
        phi_t_i = phi_t[i]
        # get those kernel points where distance is smaller than 1, so kernel > 0
        dist_smaller_1 = np.where(distances[i, :] < 1)[0]
        alpha_indices = np.array([j for j in list(dist_smaller_1)])
        alpha_indices = alpha_indices.reshape(len(alpha_indices))
        relevant_alpha = alpha[alpha_indices]
        sm = 0
        # compute sum over all alpha+kernels that have a nonzero contribution
        for idx in range(len(relevant_alpha)):
            diff = phi_t_i - kernels[idx]
            sm += relevant_alpha[idx].reshape((3, 1)) * diff.reshape((1,3)) * (1 - np.linalg.norm(diff)/c_sup)**3
            dv_dphit_i = -20/c_sup**2 * sm
        dv_dphit_diag.append(dv_dphit_i)
    return sparse.block_diag(dv_dphit_diag)


def next_du_dalpha(S, dv_dphit, prev_du_dalpha, step_size):
    m, n = S.shape
    p = step_size
    identity = sparse.identity(m)
    du_dalpha = (identity + 1/p * dv_dphit).dot(prev_du_dalpha) + 1/p * S
    return du_dalpha
