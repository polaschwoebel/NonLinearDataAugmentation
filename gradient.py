import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse
import matlab.engine
import matlab
import time
from joblib import Parallel, delayed
from joblib import load, dump
import os

#def insert_diff(phi_t, kernels, Kdev, c_sup, dim, m_c, n_c):
#    diff = phi_t[m_c, :] - kernels[n_c, :]
#    Kdev[n_c ,dim * m_c : dim * m_c + dim] = diff * (1 - np.linalg.norm(diff)
#        / c_sup) ** 3 * (-20 / (c_sup ** 2))
#
#def dv_dphit_parallel(phi_t, kernels, alpha, c_sup, dim):
#    alpha = alpha.reshape((-1, dim)).T
#
#    distances = euclidean_distances(phi_t, kernels) / c_sup
#    m, n = distances.shape
#
#    # Find evaluation- and kernel point pairs with non-zero kernel evaluations
#    dist_smaller_1 = np.where(distances < 1)
#
#    start = time.time()
#    Kdev = np.memmap("Kdev", dtype = np.float32,
#                         shape = (kernels.shape[0], dim * phi_t.shape[0]),
#                         mode='w+')
#    dump(np.copy(phi_t), "phi_t")
#    dump(np.copy(kernels), "kernels")
#
#    phi_t_dump = load("phi_t", mmap_mode = "r")
#    kernels_dump = load("kernels", mmap_mode = "r")
#
#    print("GRAD -- dump and reload ", (time.time() - start) / 60)
#    start = time.time()
#    Parallel(n_jobs=2, backend = "threading")(delayed(insert_diff)(phi_t_dump, kernels_dump,
#             Kdev, c_sup, dim, dist_smaller_1[0][i], dist_smaller_1[1][i])
#    for i in range(len(dist_smaller_1[0])))
#    print("GRAD -- run parallel ", (time.time() - start) / 60)
#
#
#    Vdev = sparse.csc_matrix.dot(sparse.csc_matrix(alpha),
#                                 sparse.csc_matrix(Kdev))
#    os.remove("Kdev")
#    os.remove("phi_t")
#    os.remove("kernels")
#    Vdev = Vdev.tolil()
#    Vdev_full = sparse.lil_matrix((dim * m, dim * m))
#    for i in range(m):
#        Vdev_full[i * dim : (i+1) * dim, i * dim : (i+1) * dim] = Vdev[:,
#                  i * dim : (i+1) * dim]
#    return Vdev_full
#
#def dv_dphit_cur(phi_t, kernels, alpha, c_sup, dim):
#    start = time.time()
#    alpha = alpha.reshape((-1, dim)).T
#
#    distances = euclidean_distances(phi_t, kernels) / c_sup
#    m, n = distances.shape
#    print("GRAD -- euc dist ", (time.time() - start) / 60)
#    # Find evaluation- and kernel point pairs with non-zero kernel evaluations
#    start = time.time()
#    dist_smaller_1 = np.where(distances < 1)
#    print("GRAD -- np.where(dist < 1) ", (time.time() - start) / 60)
#    # Compute the kernel derivative w.r.t. evaluation points
#    #Kdev = np.zeros((kernels.shape[0], dim * phi_t.shape[0]))
#    Kdev = sparse.lil_matrix((kernels.shape[0], dim * phi_t.shape[0]), dtype = np.float32)
#    start = time.time()
#    term20 = (-20 / (c_sup ** 2))
#    for i in range(len(dist_smaller_1[0])):
#
#        # Retrieve indices for non-zero kernel derivative
#        m_c = dist_smaller_1[0][i]
#        n_c = dist_smaller_1[1][i]
#        diff = phi_t[m_c, :] - kernels[n_c, :]
#                # Analytically derived Wendland kernel derivative
#        Kdev[n_c ,dim * m_c : dim * m_c + dim] = diff * (1 - np.linalg.norm(diff)
#        / c_sup) ** 3 * term20
#    print("GRAD -- Kdev insert ", (time.time() - start) / 60)
#
#    # Compute velocity derivative by multiplying alpha
#    #Vdev = sparse.csc_matrix.dot(sparse.csc_matrix(alpha), sparse.csc_matrix(Kdev))
#    start = time.time()
#    Vdev = sparse.csc_matrix.dot(sparse.csc_matrix(alpha), Kdev.tocsc())
#    print("GRAD -- alpha kdev dot ", (time.time() - start) / 60)
#    start = time.time()
#    Vdev = Vdev.tolil()
#    print("GRAD -- convert to lil ", (time.time() - start) / 60)
#    Vdev_full = sparse.lil_matrix((dim * m, dim * m))
#    start = time.time()
#    for i in range(m):
#        Vdev_full[i * dim : (i+1) * dim, i * dim : (i+1) * dim] = Vdev[:,
#                  i * dim : (i+1) * dim]
#    print("GRAD -- blowup", (time.time() - start) / 60)
#    return Vdev_full
#
#def insert_old(phi_t, kernels, dv_dphit, alpha, distances, c_sup, dim, i):
#    dist_smaller_1 = np.where(distances[i, :] < 1)[0]
#    alpha_indices = np.array([j for j in list(dist_smaller_1)])
#    sm = 0
#    # compute sum over all alpha + kernels that have a nonzero contribution
#    for alpha_idx in alpha_indices:
#        diff = phi_t[i] - kernels[alpha_idx]
#        sm += alpha[:,alpha_idx].reshape((-1, 1)).dot(diff.reshape((1,dim))) * (1 - np.linalg.norm(diff)/c_sup)**3
#    dv_dphit_i = -20/(c_sup**2) * sm
#    #dv_dphit_diag.append(dv_dphit_i)
#    dv_dphit[:, dim * i: dim * (i+1)] = dv_dphit_i
#
#def dv_dphit_old_parallel(phi_t, kernels, alpha, c_sup, dim):
#    alpha = alpha.reshape((-1, dim)).T
#    distances = euclidean_distances(phi_t, kernels)/c_sup
#    # m/3 is number of points, n/3 number of kernels
#    m, n = distances.shape
#    dv_dphit = np.memmap("dv_dphit", dtype = np.float32,
#                          shape = (dim, dim * phi_t.shape[0]),
#                          mode = "w+")
#    dump(np.copy(phi_t), "phi_t")
#    dump(np.copy(kernels), "kernels")
#
#    phi_t_dump = load("phi_t", mmap_mode = "r")
#    kernels_dump = load("kernels", mmap_mode = "r")
#
#    Parallel(n_jobs = -1, backend = "threading")(delayed(insert_old)(phi_t_dump,
#             kernels_dump, dv_dphit, alpha, distances, c_sup, dim, i)
#    for i in range(m))
#    final = sparse.block_diag(np.split(dv_dphit, m, axis = 1))
#    return final
#
#
#def dv_dphit_old(phi_t, kernels, alpha, c_sup, dim): # D_phi ("spatial jacobian") in the paper
#    alpha = alpha.reshape((-1, dim)).T
#    distances = euclidean_distances(phi_t, kernels)/c_sup
#    # m/3 is number of points, n/3 number of kernels
#    m, n = distances.shape
#    dv_dphit_diag = np.zeros((dim, dim * phi_t.shape[0]))
#    # computation of the sum of the kernel derivatives
#    # TODO: Rewrite following 2 loops as matrix multiplication?
#    for i in range(m):
#        # get those kernel points where distance is smaller than 1, so kernel > 0
#        dist_smaller_1 = np.where(distances[i, :] < 1)[0]
#        alpha_indices = np.array([j for j in list(dist_smaller_1)])
#        sm = 0
#        # compute sum over all alpha+kernels that have a nonzero contribution
#        for alpha_idx in alpha_indices:
#            diff = phi_t[i] - kernels[alpha_idx]
#            sm += alpha[:,alpha_idx].reshape((-1, 1))   .dot(diff.reshape((1,dim))) * (1 - np.linalg.norm(diff)/c_sup)**3
#        dv_dphit_i = -20/(c_sup**2) * sm
#        #dv_dphit_diag.append(dv_dphit_i)
#        dv_dphit_diag[:, dim * i: dim * (i+1)] = dv_dphit_i
#    final = sparse.block_diag(np.split(dv_dphit_diag, m, axis = 1))
#    return final
#
#def dv_dphit_parallel(phi_t, kernels, alpha, c_sup, dim):
#    alpha = alpha.reshape((-1, dim)).T
#
#    distances = euclidean_distances(phi_t, kernels) / c_sup
#    m, n = distances.shape
#
#    # Find evaluation- and kernel point pairs with non-zero kernel evaluations
#    dist_smaller_1 = np.where(distances < 1)
#
#    start = time.time()
#    Kdev = np.memmap("Kdev", dtype = np.float32,
#                         shape = (kernels.shape[0], dim * phi_t.shape[0]),
#                         mode='w+')
#    dump(np.copy(phi_t), "phi_t")
#    dump(np.copy(kernels), "kernels")
#
#    phi_t_dump = load("phi_t", mmap_mode = "r")
#    kernels_dump = load("kernels", mmap_mode = "r")
#
#    print("GRAD -- dump and reload ", (time.time() - start) / 60)
#    start = time.time()
#    Parallel(n_jobs=2, backend = "threading")(delayed(insert_diff)(phi_t_dump, kernels_dump,
#             Kdev, c_sup, dim, dist_smaller_1[0][i], dist_smaller_1[1][i])
#    for i in range(len(dist_smaller_1[0])))
#    print("GRAD -- run parallel ", (time.time() - start) / 60)
#
#
#    Vdev = sparse.csc_matrix.dot(sparse.csc_matrix(alpha),
#                                 sparse.csc_matrix(Kdev))
#    os.remove("Kdev")
#    os.remove("phi_t")
#    os.remove("kernels")
#    Vdev = Vdev.tolil()
#    Vdev_full = sparse.lil_matrix((dim * m, dim * m))
#    for i in range(m):
#        Vdev_full[i * dim : (i+1) * dim, i * dim : (i+1) * dim] = Vdev[:,
#                  i * dim : (i+1) * dim]
#    return Vdev_full


def dv_dphit(phi_t, kernels, alpha, c_sup, dim):
    alpha = alpha.reshape((-1, dim)).T
    distances = euclidean_distances(phi_t, kernels)/c_sup
    # m is number of points, n number of kernels
    m, n = distances.shape
    dv_dphit_diag = sparse.lil_matrix((dim * phi_t.shape[0], dim * phi_t.shape[0]))
    # computation of the sum of the kernel derivatives
    factor = -20/(c_sup**2)
    for i in range(m):
        # get those kernel points where distance is smaller than 1, so kernel > 0
        dist_smaller_1 = np.where(distances[i, :] < 1)[0]
        alpha_indices = np.array([j for j in list(dist_smaller_1)])
        sm = 0
        # compute sum over all alpha+kernels that have a nonzero contribution
        for alpha_idx in alpha_indices:
            diff = phi_t[i] - kernels[alpha_idx]
            sm += alpha[:,alpha_idx].reshape((-1, 1)).dot(diff.reshape((1,dim))) * (1 - np.linalg.norm(diff)/c_sup)**3
        dv_dphit_i = factor * sm
        dv_dphit_diag[dim * i: dim * (i+1), dim * i: dim * (i+1)] = dv_dphit_i
    return dv_dphit_diag

def dv_dphit_old_parallel(phi_t, kernels, alpha, c_sup, dim): # D_phi ("spatial jacobian") in the paper
    #start = time.time()
    dim = kernels.shape[1]
    alpha = alpha.reshape((-1,dim))
    distances = euclidean_distances(phi_t, kernels)/c_sup
    #print("GRAD -- euc dist ", (time.time() - start) / 60)
    # m is number of points, n number of kernels
    m, n = distances.shape
    print(m, n)
    #dv_dphit_diag = np.empty((m, dim, dim))
    dv_dphit_diag = np.memmap("dv_dphit_diag", dtype = np.float32,
                         shape = (dim*m, dim*m),
                         mode='w+')
    dump(np.copy(phi_t), "phi_t")
    dump(np.copy(kernels), "kernels")
    dump(np.copy(alpha), "alpha")
    phi_t_dump = load("phi_t", mmap_mode = "r")
    kernels_dump = load("kernels", mmap_mode = "r")
    alpha_dump = load("alpha", mmap_mode = "r")
     # computation of the sum of the kernel derivatives
    #start_loop = time.time()
    Parallel(n_jobs=24, backend = "threading")(delayed(compute_block)(phi_t_dump, kernels_dump,
        alpha_dump, dv_dphit_diag, distances[i,:], c_sup, dim, i) for i in range(m))
    #print("GRAD -- for loop done", (time.time() - start_loop) / 60)
    os.remove("alpha")
    os.remove("phi_t")
    os.remove("kernels")
    before_res = time.time()
    #result = sparse.block_diag(dv_dphit_diag)
    #print("GRAD -- made into sparse", (time.time() - before_res) / 60)
    #print("GRAD -- total", (time.time() - start) / 60)
     #return result
    return sparse.csc_matrix(dv_dphit_diag)


def compute_block(phi_t, kernels, alpha, dv_dphit_diag, distances, c_sup, dim, i):
    phi_t_i = phi_t[i]
    # get those kernel points where distance is smaller than 1, so kernel > 0
    dist_smaller_1 = np.where(distances < 1)[0]
    alpha_indices = np.array([j for j in list(dist_smaller_1)])
    sm = 0
    factor = -20/(c_sup**2)
    # compute sum over all alpha+kernels that have a nonzero contribution
    for alpha_idx in alpha_indices:
        diff = phi_t_i - kernels[alpha_idx]
        sm += alpha[alpha_idx].reshape((dim,1)).dot(diff.reshape((1,dim))) * (1 - np.linalg.norm(diff)/c_sup)**3
    dv_dphit_i = factor * sm
    dv_dphit_diag[dim*i : dim*i + dim, dim*i : dim*i + dim] = dv_dphit_i

# Compute dphi_dalpha for next Forward Euler step by the recursive definition
def next_dphi_dalpha(S, dv_dphit, prev_dphi_dalpha, step_size):
    m, n = S.shape
    identity = sparse.identity(m)
    dphi_dalpha = (identity + 1/step_size * dv_dphit).dot(prev_dphi_dalpha) + 1/step_size * S
    return dphi_dalpha

# Computation of the image gradient. Returns the full Jacobian matrix of
# dimension 3m x 3m where m is number of evaluation points
def dIm_dphi(img, eng, spline_rep, phi, res, dim):
    if (dim == 2):
        phi_x = matlab.double(phi[:,0].tolist())
        phi_y = matlab.double(phi[:,1].tolist())

        # Use spline representation of image to extract derivatives at phi
        imres = img.shape[0]
        dev1 = np.array(eng.eval_dev12d(spline_rep, phi_x, phi_y, imres), dtype=np.float32)
        dev2 = np.array(eng.eval_dev22d(spline_rep, phi_x, phi_y, imres), dtype=np.float32)
        dev1[np.isnan(dev1)] = 0
        dev2[np.isnan(dev2)] = 0
        gradients_all_dims = [dev2, dev1]
        gradient_array = np.dstack([dim_arr.flatten(order='F') for dim_arr in gradients_all_dims])[::-1][0]
        d = sparse.block_diag(gradient_array)
    else:
        phi_x = matlab.double(phi[:,0].tolist())
        phi_y = matlab.double(phi[:,1].tolist())
        phi_z = matlab.double(phi[:,2].tolist())
        # Use spline representation of image to extract derivatives at phi
        (xres, yres, zres) = img.shape
        dev1 = np.array(eng.eval_dev13d(spline_rep, phi_x, phi_y, phi_z ,
                                        xres, yres, zres), dtype=np.float32)
        dev2 = np.array(eng.eval_dev23d(spline_rep, phi_x, phi_y, phi_z ,
                                        xres, yres, zres), dtype=np.float32)
        dev3 = np.array(eng.eval_dev33d(spline_rep, phi_x, phi_y, phi_z ,
                                        xres, yres, zres), dtype=np.float32)
        dev1[np.isnan(dev1)] = 0
        dev2[np.isnan(dev2)] = 0
        dev3[np.isnan(dev3)] = 0

        m = phi.shape[0]
        d = sparse.coo_matrix((m, dim * m))
        d.data = np.concatenate((dev2.flatten(order='F'),
                                 dev1.flatten(order='F'),
                                 dev3.flatten(order='F')), axis = 0)
        row = np.arange(m)
        col = np.arange(0, 3 * m, dim)
        d.row = np.concatenate((row, row, row))
        d.col = np.concatenate((col, col + 1, col + 2), axis = 0)

        #gradients_all_dims = [dev2, dev1, dev3]
        #gradient_array = np.dstack([dim_arr.flatten(order='F') for dim_arr in gradients_all_dims])[::-1][0]
        #block_diag = sparse.block_diag(gradient_array)
    return d

def dED_dphit(im_source, im_target, trans_points, points, dIm1_dphi1, dim):
    if dim == 3:
        target_points = im_target[points[:, 1], points[:, 0], points[:, 2]]
    else:
        target_points = im_target[points[:, 1], points[:, 0]]
    diff = sparse.csr_matrix(trans_points - target_points)
    return sparse.csc_matrix(2 * diff.dot(dIm1_dphi1))

def dER_dalpha(G, alpha):
    alpha = alpha.reshape((-1, 1))
    return sparse.csc_matrix(2 * G.dot(alpha))
