#%%
import numpy as np
import scipy.ndimage.morphology as scm
import os
from scipy.io import loadmat, savemat
from dipy.align import imaffine, transforms

#%%
#### PRE PROCESSING ####

# Load images/labels and find common dimensions such that all images are stored
# in a single tensor, and all labels are stored in a single tensor
def align_dimensions(data_path, label_path):    
    data_files = os.listdir(data_path)
    max_depth = 0
    
    print("ALIGN DIMENSIONS: processing images")
    for file in data_files:
        (h, w, mat_depth) = loadmat(data_path + file)["data"].shape
        max_depth = mat_depth if mat_depth > max_depth else max_depth
    
    data = np.zeros((len(data_files), h, w, max_depth), dtype = np.uint16)
    for i in range(0, len(data_files)):
        file = data_files[i]
        mat = loadmat(data_path + file)["data"]
        data[i, :, :, :mat.shape[2]] = mat
    
    print("ALIGN DIMENSIONS: processing labels")
    label_files = os.listdir(label_path)
    labels = np.zeros((len(label_files), h, w, max_depth), dtype = np.uint8)
    for i in range(0, len(label_files)):
        file = label_files[i]
        mat = loadmat(label_path + file)["label"]
        labels[i, :, :, :mat.shape[2]] = mat
    
    return (data, labels)

# Find non-background mask for single image
def find_skull_strip_mask(labels):
    m = np.zeros_like(labels)
    m = scm.binary_dilation(labels, iterations = 1)
    m = scm.binary_fill_holes(m)
    return m.astype(np.uint8)

# Set to background (0) all pixels outside mask
def filter_im(im, mask):
    im_noskull = np.copy(im)
    im_noskull[np.invert(mask == 1)] = 0
    return im_noskull

# Remove skulls based in labels
def remove_skulls(data, labels):
    newdata = np.zeros_like(data)
    for i in range(data.shape[0]):
        print("REMOVE SKULLS: processing image: ", i)
        mask = find_skull_strip_mask(labels[i,:,:,:])
        newdata[i,:,:,:] = filter_im(data[i,:,:,:], mask)
    return newdata

# Normalize image intensities individually
def normalize_intensity(data):
    (b, h, w, d) = data.shape
    data_norm = np.zeros_like(data).astype(np.float32)
    for i in range(b):
        I = data[i,:,:,:]
        mean = np.mean(I)
        std = np.std(I)
        data_norm[i,:,:,:] = np.divide(np.subtract(I, mean), std)
    return data_norm

def register_affinely(static, moving):
    affine_reg = imaffine.AffineRegistration(metric=None, level_iters=[250, 10], sigmas=None,
                    factors=None, method='L-BFGS-B', ss_sigma_factor=None, options=None, verbosity=2)
    transform =  transforms.AffineTransform3D()
    params0 = None
    affine_map = affine_reg.optimize(static, moving, transform, params0)
    return affine_map

def transform(moving, affine_map):
    moving = moving.astype(float)
    transformed_image = affine_map.transform(moving, interp='nearest')
    return transformed_image

# Affinely aligns all images in data to the first image in data. 
# Transforms corresponding labels as well
def affine_align_all_data(data, labels):
    aligned_data = np.zeros_like(data)
    aligned_labels = np.zeros_like(labels)
    for i in range(data.shape[0]):
        print("AFFINE ALIGN: processing image ", i)
        affine_reg = register_affinely(data[0,:,:,:], data[i,:,:,:])
        aligned_data[i,:,:,:] = transform(data[i,:,:,:], affine_reg)
        aligned_labels[i,:,:,:] = transform(labels[i,:,:,:], affine_reg)
    return (aligned_data, aligned_labels)

def padding_to_normalized_background(data):
    data2 = np.copy(data)
    for i in range(data2.shape[0]):
        un = np.unique(data2[i,:,:,:])
        data2[i,:,:,:][data2[i,:,:,:] == 0] = un[0]
    return data2

def run_whole_preprocessing(data_path, labels_path):
    (data, labels) = align_dimensions(data_path, labels_path)
    
    data = remove_skulls(data, labels)
    data = normalize_intensity(data)
    
    (data, labels) = affine_align_all_data(data, labels)
    data = padding_to_normalized_background(data)
    return (data, labels)


#### REGISTRATION ####

# Finds mask that contains all non-background pixels over all images.
# Used for specifying relevant evaluation points
def find_relevant_points_mask(labels, dilations):
    mask = np.zeros_like(labels[0])
    for i in range(len(labels)):
        m = scm.binary_dilation(labels[i], iterations = dilations)
        m = scm.binary_fill_holes(m)
        mask = np.logical_or(mask, m)
    return m.astype(np.uint8)