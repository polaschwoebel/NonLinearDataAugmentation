import matlab.engine
import numpy as np
import registration
import pre_processing
import diffutils as utils
import time

debug = True

def main():
    start = time.time()
    if not debug:
        # Martin
        affine_images_path = 'MICCAI_data/train_ims_aligned_noskull.npy'
        affine_labels_path = 'MICCAI_data/train_ims_aligned_noskull.npy'
        # Pola
        affine_images = np.load(affine_images_path)
        affine_labels = np.load(affine_labels_path)

        im1 = affine_images[0, :, :, 100]
        im2 = affine_images[5, :, :, 100]
        labels1 = affine_labels[0, :, :, :]
        labels2 = affine_labels[5, :, :, :]

    if debug:
        # lowres for debugging
        im1 = np.load('../MICCAI_data/lowres/I1_low.npy')
        im2 = np.load('../MICCAI_data/lowres/I5_low.npy')
        labels1 = np.load('../MICCAI_data/lowres/L1_low.npy')
        labels2 = np.load('../MICCAI_data/lowres/L5_low.npy')



    options = {}
    options['c_sup'] = 16
    options['kernel_res'] = 8
    options['eval_res'] = 2
    options['dim'] = 3
    options['opt_eps'] = 0.02
    options['opt_tol'] = 0.01
    options['reg_weight'] = 0.002
    options['opt_maxiter'] = 30
    eval_mask = pre_processing.find_relevant_points_mask([labels1, labels2], 2)
    kernel_mask = pre_processing.find_relevant_points_mask([labels1, labels2], options['c_sup'])
    options['kernel_mask'] = kernel_mask
    options['eval_mask'] = eval_mask

    alpha = registration.find_transformation(im1, im2, options)
    end = time.time()
    print('Registration done. Time:', end - start)

    np.save('alpha_0_5.npy', alpha)
    reconstructed = utils.apply_trafo_full(im1, im2, options)
    np.save('reconstructed_0_5.npy', reconstructed)


if __name__ == main():
    main()
