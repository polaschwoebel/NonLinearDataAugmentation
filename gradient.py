import matlab.engine
import matlab
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse
import time
import tempfile
import shutil
from joblib import Parallel, delayed
from joblib import load, dump
import os


def dv_dphit_parallel(phi_t, kernels, alpha, c_sup, dim): # D_phi ("spatial jacobian") in the paper
    start = time.time()
    dim = kernels.shape[1]
    alpha = alpha.reshape((-1,dim))
    distances = euclidean_distances(phi_t, kernels)/c_sup
    #print("GRAD -- euc dist ", (time.time() - start) / 60)
    # m is number of points, n number of kernels
    m, n = distances.shape
    #dv_dphit_diag = np.empty((m, dim, dim))
    dv_dphit_diag = np.memmap("dv_dphit_diag", dtype = np.float32,
                         shape=(dim*m, dim*m),
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

def dv_dphit(phi_t, kernels, alpha, c_sup, dim): # D_phi ("spatial jacobian") in the paper
    alpha = alpha.reshape((-1, dim)).T
    distances = euclidean_distances(phi_t, kernels)/c_sup
    # m/3 is number of points, n/3 number of kernels
    m, n = distances.shape
    dv_dphit_diag = sparse.lil_matrix((dim * phi_t.shape[0], dim * phi_t.shape[0]))
    # computation of the sum of the kernel derivatives
    # TODO: Rewrite following 2 loops as matrix multiplication?
    start = time.time()
    for i in range(m):
        # get those kernel points where distance is smaller than 1, so kernel > 0
        dist_smaller_1 = np.where(distances[i, :] < 1)[0]
        alpha_indices = np.array([j for j in list(dist_smaller_1)])
        sm = 0
        # compute sum over all alpha+kernels that have a nonzero contribution
        for alpha_idx in alpha_indices:
            diff = phi_t[i] - kernels[alpha_idx]
            sm += alpha[:,alpha_idx].reshape((-1, 1)).dot(diff.reshape((1,dim))) * (1 - np.linalg.norm(diff)/c_sup)**3
        dv_dphit_i = -20/(c_sup**2) * sm
        #dv_dphit_diag.append(dv_dphit_i)
        dv_dphit_diag[dim * i: dim * (i+1), dim * i: dim * (i+1)] = dv_dphit_i
    #print("GRAD old sparse -- loop ", (time.time() - start) / 60)
    #final = sparse.block_diag(np.split(dv_dphit_diag, m, axis = 1))
    return dv_dphit_diag

# Compute dphi_dalpha for next Forward Euler step by the recursive definition
def next_dphi_dalpha(S, dv_dphit, prev_dphi_dalpha, step_size, dim):
    m, n = S.shape
    identity = sparse.identity(m)
    dphi_dalpha = (identity + 1/step_size * dv_dphit).dot(prev_dphi_dalpha) + 1/step_size * S
    return dphi_dalpha.tocsc()

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
        block_diag = sparse.block_diag(gradient_array)
    else:
        start = time.time()
        phi_x = matlab.double(phi[:,0].tolist())
        phi_y = matlab.double(phi[:,1].tolist())
        phi_z = matlab.double(phi[:,2].tolist())
        # Use spline representation of image to extract derivatives at phi
        (xres, yres, zres) = img.shape
        start = time.time()
        dev1 = np.array(eng.eval_dev13d(spline_rep, phi_x, phi_y, phi_z ,
                                        xres, yres, zres), dtype=np.float32)
        dev2 = np.array(eng.eval_dev23d(spline_rep, phi_x, phi_y, phi_z ,
                                        xres, yres, zres), dtype=np.float32)
        dev3 = np.array(eng.eval_dev33d(spline_rep, phi_x, phi_y, phi_z ,
                                        xres, yres, zres), dtype=np.float32)
        start = time.time()
        dev1[np.isnan(dev1)] = 0
        dev2[np.isnan(dev2)] = 0
        dev3[np.isnan(dev3)] = 0
        start = time.time()
        gradients_all_dims = [dev2, dev1, dev3]
        start = time.time()
        m = phi.shape[0]
        d = sparse.coo_matrix((m, dim * m))
        d.data = np.concatenate((dev2.flatten(order='F'), dev1.flatten(order='F'), dev3.flatten(order='F')), axis = 0)
        row = np.arange(m)
        col = np.arange(0, 3 * m, dim)
        d.row = np.concatenate((row, row, row), axis = 0)
        d.col = np.concatenate((col, col + 1, col + 2), axis = 0)
        #gradient_array = np.dstack([dim_arr.flatten(order='F') for dim_arr in gradients_all_dims])[::-1][0]
        #print("GRAD dI_dphi -- dstack ", (time.time() - start) / 60)
        start = time.time()
        #block_diag = sparse.block_diag(gradient_array)
        #print("GRAD dI_dphi -- block diag ", (time.time() - start) / 60)
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

