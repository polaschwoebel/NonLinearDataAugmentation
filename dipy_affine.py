from dipy.align import imaffine, transforms
from scipy import io as scio
import numpy as np


def register_affinely(static, moving):
    affine_reg = imaffine.AffineRegistration(metric=None, level_iters=[250, 10], sigmas=None,
                    factors=None, method='L-BFGS-B', ss_sigma_factor=None, options=None, verbosity=1)
    transform =  transforms.AffineTransform3D()
    params0 = None
    affine_map = affine_reg.optimize(static, moving, transform, params0)
    return affine_map


def transform(moving, affine_map):
    moving = moving.astype(float)
    transformed_image = affine_map.transform(moving, interp='nearest')
    return transformed_image


train = scio.loadmat('../miccai/miccai_train.mat')
train_data = train['data']
train_labels = train['labels']
nr_images = train_data.shape[0]
target_img = train_data[0,:,:,:]

test = scio.loadmat('../miccai/miccai_test.mat')
test_data = test['data']
test_labels = test['labels']
nr_images = test_data.shape[0]
target_img = test_data[0,:,:,:]

if __name__== "__main__":
    for ix in range(7, nr_images):
        print('Processing test image', ix)
        moving_img = test_data[ix,:,:,:]
        moving_labels = test_labels[ix,:,:,:]

        affine_reg = register_affinely(target_img, moving_img)
        transformed_img = transform(moving_img, affine_reg)
        transformed_labels = transform(moving_labels, affine_reg)

        np.save('../miccai/affinely_aligned/test_img/img_%s.npy' %ix, transformed_img, allow_pickle=False)
        np.save('../miccai/affinely_aligned/test_labels/label_%s.npy' %ix, transformed_labels, allow_pickle=False)
