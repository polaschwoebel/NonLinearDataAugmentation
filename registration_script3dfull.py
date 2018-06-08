import matlab.engine
import matlab
import registration
import numpy as np
import pre_processing
import diffutils as utils
import time


start = time.time()
#affine_images_path = '../miccai_data_noskull/train_ims_aligned_noskull.npy'
#affine_labels_path = '../miccai_data_noskull/train_labs_aligned_noskull.npy'
#affine_images = np.load(affine_images_path)
#affine_labels = np.load(affine_labels_path)

#im1 = affine_images[0, :, :, :]
#im2 = affine_images[5, :, :, :]

#labels1 = affine_labels[0, :, :, :]
#labels2 = affine_labels[5, :, :, :]

#im1 = np.load("I1_low.npy")
#im2 = np.load("I5_low.npy")
#labels1 = np.load("L1_low.npy")
#labels2 = np.load("L5_low.npy")

im1 = np.load("I1.npy")
im2 = np.load("I5.npy")
labels1 = np.load("L1.npy")
labels2 = np.load("L5.npy")

options = {}
options['c_sup'] = 12
options['kernel_res'] = 6
options['eval_res'] = 3
options['dim'] = 3
options['opt_eps'] = 0.04
options['opt_tol'] = 0.01
options['reg_weight'] = 0.2
options['opt_maxiter'] = 20
eval_mask = pre_processing.find_relevant_points_mask([labels1, labels2], 2)
kernel_mask = pre_processing.find_relevant_points_mask([labels1, labels2], options['c_sup'])
options['kernel_mask'] = kernel_mask
options['eval_mask'] = eval_mask

alpha = registration.find_transformation(im1, im2, options)
end = time.time()
print('Registration done. Time:', end - start)

np.save('alpha_0_5_it20eps004reg02.npy', alpha)
#alpha = np.load("alpha_0_5.npy")
#reconstructed = utils.apply_trafo_full(im1, alpha, options)
#np.save('reconstructed_0_5_3d.npy', reconstructed)
